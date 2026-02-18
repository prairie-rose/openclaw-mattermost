import type {
  ChannelAccountSnapshot,
  ChatType,
  OpenClawConfig,
  ReplyPayload,
  RuntimeEnv,
} from "openclaw/plugin-sdk";
import {
  buildAgentMediaPayload,
  createReplyPrefixOptions,
  createTypingCallbacks,
  logInboundDrop,
  logTypingFailure,
  buildPendingHistoryContextFromMap,
  clearHistoryEntriesIfEnabled,
  DEFAULT_GROUP_HISTORY_LIMIT,
  recordPendingHistoryEntryIfEnabled,
  resolveControlCommandGate,
  resolveChannelMediaMaxBytes,
  type HistoryEntry,
} from "openclaw/plugin-sdk";
import * as fs from "node:fs/promises";

/** ImageContent type for passing images to the LLM context. */
type ImageContent = {
  type: "image";
  data: string; // base64 encoded image data
  mimeType: string; // e.g., "image/jpeg", "image/png"
};

/**
 * Check if a URL points to a private/local network address.
 * Used to conditionally allow SSRF for self-hosted Mattermost instances.
 */
function isPrivateNetworkUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();
    if (host === "localhost" || host === "127.0.0.1" || host === "::1") {
      return true;
    }
    const ipv4Match = host.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
    if (ipv4Match) {
      const [, a, b] = ipv4Match.map(Number);
      if (a === 10) return true;
      if (a === 172 && b >= 16 && b <= 31) return true;
      if (a === 192 && b === 168) return true;
    }
    return false;
  } catch {
    return false;
  }
}

import { getMattermostRuntime } from "../runtime.js";
import { resolveMattermostAccount } from "./accounts.js";
import {
  createMattermostClient,
  fetchMattermostChannel,
  fetchMattermostMe,
  fetchMattermostUser,
  normalizeMattermostBaseUrl,
  sendMattermostTyping,
  type MattermostChannel,
  type MattermostPost,
  type MattermostUser,
} from "./client.js";
import {
  createDedupeCache,
  formatInboundFromLabel,
  normalizeMention,
  resolveThreadSessionKeys,
} from "./monitor-helpers.js";
import { resolveOncharPrefixes, stripOncharPrefix } from "./monitor-onchar.js";
import {
  createMattermostConnectOnce,
  type MattermostEventPayload,
  type MattermostWebSocketFactory,
} from "./monitor-websocket.js";
import { runWithReconnect } from "./reconnect.js";
import { sendMessageMattermost } from "./send.js";

export type MonitorMattermostOpts = {
  botToken?: string;
  baseUrl?: string;
  accountId?: string;
  config?: OpenClawConfig;
  runtime?: RuntimeEnv;
  abortSignal?: AbortSignal;
  statusSink?: (patch: Partial<ChannelAccountSnapshot>) => void;
  webSocketFactory?: MattermostWebSocketFactory;
};

type FetchLike = (input: URL | RequestInfo, init?: RequestInit) => Promise<Response>;
type MediaKind = "image" | "audio" | "video" | "document" | "unknown";

const RECENT_MATTERMOST_MESSAGE_TTL_MS = 5 * 60_000;
const RECENT_MATTERMOST_MESSAGE_MAX = 2000;
const CHANNEL_CACHE_TTL_MS = 5 * 60_000;
const USER_CACHE_TTL_MS = 10 * 60_000;

const recentInboundMessages = createDedupeCache({
  ttlMs: RECENT_MATTERMOST_MESSAGE_TTL_MS,
  maxSize: RECENT_MATTERMOST_MESSAGE_MAX,
});

function resolveRuntime(opts: MonitorMattermostOpts): RuntimeEnv {
  return (
    opts.runtime ?? {
      log: console.log,
      error: console.error,
      exit: (code: number): never => {
        throw new Error(`exit ${code}`);
      },
    }
  );
}

function isSystemPost(post: MattermostPost): boolean {
  const type = post.type?.trim();
  return Boolean(type);
}

function channelKind(channelType?: string | null): ChatType {
  if (!channelType) {
    return "channel";
  }
  const normalized = channelType.trim().toUpperCase();
  if (normalized === "D") {
    return "direct";
  }
  if (normalized === "G") {
    return "group";
  }
  return "channel";
}

function channelChatType(kind: ChatType): "direct" | "group" | "channel" {
  if (kind === "direct") {
    return "direct";
  }
  if (kind === "group") {
    return "group";
  }
  return "channel";
}

function normalizeAllowEntry(entry: string): string {
  const trimmed = entry.trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed === "*") {
    return "*";
  }
  return trimmed
    .replace(/^(mattermost|user):/i, "")
    .replace(/^@/, "")
    .toLowerCase();
}

function normalizeAllowList(entries: Array<string | number>): string[] {
  const normalized = entries.map((entry) => normalizeAllowEntry(String(entry))).filter(Boolean);
  return Array.from(new Set(normalized));
}

function isSenderAllowed(params: {
  senderId: string;
  senderName?: string;
  allowFrom: string[];
}): boolean {
  const allowFrom = params.allowFrom;
  if (allowFrom.length === 0) {
    return false;
  }
  if (allowFrom.includes("*")) {
    return true;
  }
  const normalizedSenderId = normalizeAllowEntry(params.senderId);
  const normalizedSenderName = params.senderName ? normalizeAllowEntry(params.senderName) : "";
  return allowFrom.some(
    (entry) =>
      entry === normalizedSenderId || (normalizedSenderName && entry === normalizedSenderName),
  );
}

type MattermostMediaInfo = {
  path: string;
  contentType?: string;
  kind: MediaKind;
};

function buildMattermostAttachmentPlaceholder(mediaList: MattermostMediaInfo[]): string {
  if (mediaList.length === 0) {
    return "";
  }
  if (mediaList.length === 1) {
    const kind = mediaList[0].kind === "unknown" ? "document" : mediaList[0].kind;
    return `<media:${kind}>`;
  }
  const allImages = mediaList.every((media) => media.kind === "image");
  const label = allImages ? "image" : "file";
  const suffix = mediaList.length === 1 ? label : `${label}s`;
  const tag = allImages ? "<media:image>" : "<media:document>";
  return `${tag} (${mediaList.length} ${suffix})`;
}

/**
 * Converts image media files to ImageContent array for passing to the LLM.
 * Skips images that exceed MAX_IMAGE_BYTES to avoid memory issues with base64 expansion.
 */
const MAX_IMAGE_BYTES_FOR_LLM = 5 * 1024 * 1024; // 5MB limit before base64 encoding

async function buildMattermostImageContents(
  mediaList: MattermostMediaInfo[],
  logger?: { debug?: (msg: string) => void },
): Promise<ImageContent[]> {
  const images: ImageContent[] = [];
  const imageMediaList = mediaList.filter(
    (media) => media.kind === "image" && media.contentType?.startsWith("image/"),
  );

  for (const media of imageMediaList) {
    try {
      const stats = await fs.stat(media.path);
      if (stats.size > MAX_IMAGE_BYTES_FOR_LLM) {
        logger?.debug?.(
          `mattermost: skipping oversized image ${media.path} (${stats.size} bytes > ${MAX_IMAGE_BYTES_FOR_LLM})`,
        );
        continue;
      }
      const buffer = await fs.readFile(media.path);
      const base64 = buffer.toString("base64");
      images.push({
        type: "image",
        data: base64,
        mimeType: media.contentType ?? "image/jpeg",
      });
    } catch (err) {
      logger?.debug?.(`mattermost: failed to read image file ${media.path}: ${String(err)}`);
    }
  }

  return images;
}

function buildMattermostWsUrl(baseUrl: string): string {
  const normalized = normalizeMattermostBaseUrl(baseUrl);
  if (!normalized) {
    throw new Error("Mattermost baseUrl is required");
  }
  const wsBase = normalized.replace(/^http/i, "ws");
  return `${wsBase}/api/v4/websocket`;
}

export async function monitorMattermostProvider(opts: MonitorMattermostOpts = {}): Promise<void> {
  const core = getMattermostRuntime();
  const runtime = resolveRuntime(opts);
  const cfg = opts.config ?? core.config.loadConfig();
  const account = resolveMattermostAccount({
    cfg,
    accountId: opts.accountId,
  });
  const botToken = opts.botToken?.trim() || account.botToken?.trim();
  if (!botToken) {
    throw new Error(
      `Mattermost bot token missing for account "${account.accountId}" (set channels.mattermost.accounts.${account.accountId}.botToken or MATTERMOST_BOT_TOKEN for default).`,
    );
  }
  const baseUrl = normalizeMattermostBaseUrl(opts.baseUrl ?? account.baseUrl);
  if (!baseUrl) {
    throw new Error(
      `Mattermost baseUrl missing for account "${account.accountId}" (set channels.mattermost.accounts.${account.accountId}.baseUrl or MATTERMOST_URL for default).`,
    );
  }

  const client = createMattermostClient({ baseUrl, botToken });
  const botUser = await fetchMattermostMe(client);
  const botUserId = botUser.id;
  const botUsername = botUser.username?.trim() || undefined;
  runtime.log?.(`mattermost connected as ${botUsername ? `@${botUsername}` : botUserId}`);

  const channelCache = new Map<string, { value: MattermostChannel | null; expiresAt: number }>();
  const userCache = new Map<string, { value: MattermostUser | null; expiresAt: number }>();
  const logger = core.logging.getChildLogger({ module: "mattermost" });
  const logVerboseMessage = (message: string) => {
    if (!core.logging.shouldLogVerbose()) {
      return;
    }
    logger.debug?.(message);
  };
  const mediaMaxBytes =
    resolveChannelMediaMaxBytes({
      cfg,
      resolveChannelLimitMb: () => undefined,
      accountId: account.accountId,
    }) ?? 8 * 1024 * 1024;
  const historyLimit = Math.max(
    0,
    cfg.messages?.groupChat?.historyLimit ?? DEFAULT_GROUP_HISTORY_LIMIT,
  );
  const channelHistories = new Map<string, HistoryEntry[]>();

  const fetchWithAuth: FetchLike = (input, init) => {
    const headers = new Headers(init?.headers);
    headers.set("Authorization", `Bearer ${client.token}`);
    return fetch(input, { ...init, headers });
  };

  const resolveMattermostMedia = async (
    fileIds?: string[] | null,
  ): Promise<MattermostMediaInfo[]> => {
    const ids = (fileIds ?? []).map((id) => id?.trim()).filter(Boolean);
    if (ids.length === 0) {
      return [];
    }
    const out: MattermostMediaInfo[] = [];
    for (const fileId of ids) {
      try {
        // Only allow private network SSRF for self-hosted instances (localhost/private IPs)
        const allowPrivate = isPrivateNetworkUrl(client.apiBaseUrl);
        const fetched = await core.channel.media.fetchRemoteMedia({
          url: `${client.apiBaseUrl}/files/${fileId}`,
          fetchImpl: fetchWithAuth,
          filePathHint: fileId,
          maxBytes: mediaMaxBytes,
          ...(allowPrivate ? { ssrfPolicy: { allowPrivateNetwork: true } } : {}),
        });
        const saved = await core.channel.media.saveMediaBuffer(
          fetched.buffer,
          fetched.contentType ?? undefined,
          "inbound",
          mediaMaxBytes,
        );
        const contentType = saved.contentType ?? fetched.contentType ?? undefined;
        out.push({
          path: saved.path,
          contentType,
          kind: core.media.mediaKindFromMime(contentType),
        });
      } catch (err) {
        logger.debug?.(`mattermost: failed to download file ${fileId}: ${String(err)}`);
      }
    }
    return out;
  };

  const sendTypingIndicator = async (channelId: string, parentId?: string) => {
    await sendMattermostTyping(client, { channelId, parentId });
  };

  const resolveChannelInfo = async (channelId: string): Promise<MattermostChannel | null> => {
    const cached = channelCache.get(channelId);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value;
    }
    try {
      const info = await fetchMattermostChannel(client, channelId);
      channelCache.set(channelId, {
        value: info,
        expiresAt: Date.now() + CHANNEL_CACHE_TTL_MS,
      });
      return info;
    } catch (err) {
      logger.debug?.(`mattermost: channel lookup failed: ${String(err)}`);
      channelCache.set(channelId, {
        value: null,
        expiresAt: Date.now() + CHANNEL_CACHE_TTL_MS,
      });
      return null;
    }
  };

  const resolveUserInfo = async (userId: string): Promise<MattermostUser | null> => {
    const cached = userCache.get(userId);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value;
    }
    try {
      const info = await fetchMattermostUser(client, userId);
      userCache.set(userId, {
        value: info,
        expiresAt: Date.now() + USER_CACHE_TTL_MS,
      });
      return info;
    } catch (err) {
      logger.debug?.(`mattermost: user lookup failed: ${String(err)}`);
      userCache.set(userId, {
        value: null,
        expiresAt: Date.now() + USER_CACHE_TTL_MS,
      });
      return null;
    }
  };

  const handlePost = async (
    post: MattermostPost,
    payload: MattermostEventPayload,
    messageIds?: string[],
  ) => {
    const channelId = post.channel_id ?? payload.data?.channel_id ?? payload.broadcast?.channel_id;
    if (!channelId) {
      return;
    }

    const allMessageIds = messageIds?.length ? messageIds : post.id ? [post.id] : [];
    if (allMessageIds.length === 0) {
      return;
    }
    const dedupeEntries = allMessageIds.map((id) =>
      recentInboundMessages.check(`${account.accountId}:${id}`),
    );
    if (dedupeEntries.length > 0 && dedupeEntries.every(Boolean)) {
      return;
    }

    const senderId = post.user_id ?? payload.broadcast?.user_id;
    if (!senderId) {
      return;
    }
    if (senderId === botUserId) {
      return;
    }
    if (isSystemPost(post)) {
      return;
    }

    const channelInfo = await resolveChannelInfo(channelId);
    const channelType = payload.data?.channel_type ?? channelInfo?.type ?? undefined;
    const kind = channelKind(channelType);
    const chatType = channelChatType(kind);

    const senderName =
      payload.data?.sender_name?.trim() ||
      (await resolveUserInfo(senderId))?.username?.trim() ||
      senderId;
    const rawText = post.message?.trim() || "";
    const dmPolicy = account.config.dmPolicy ?? "pairing";
    const defaultGroupPolicy = cfg.channels?.defaults?.groupPolicy;
    const groupPolicy = account.config.groupPolicy ?? defaultGroupPolicy ?? "allowlist";
    const configAllowFrom = normalizeAllowList(account.config.allowFrom ?? []);
    const configGroupAllowFrom = normalizeAllowList(account.config.groupAllowFrom ?? []);
    const storeAllowFrom = normalizeAllowList(
      await core.channel.pairing.readAllowFromStore("mattermost").catch(() => []),
    );
    const effectiveAllowFrom = Array.from(new Set([...configAllowFrom, ...storeAllowFrom]));
    const effectiveGroupAllowFrom = Array.from(
      new Set([
        ...(configGroupAllowFrom.length > 0 ? configGroupAllowFrom : configAllowFrom),
        ...storeAllowFrom,
      ]),
    );
    const allowTextCommands = core.channel.commands.shouldHandleTextCommands({
      cfg,
      surface: "mattermost",
    });
    const hasControlCommand = core.channel.text.hasControlCommand(rawText, cfg);
    const isControlCommand = allowTextCommands && hasControlCommand;
    const useAccessGroups = cfg.commands?.useAccessGroups !== false;
    const senderAllowedForCommands = isSenderAllowed({
      senderId,
      senderName,
      allowFrom: effectiveAllowFrom,
    });
    const groupAllowedForCommands = isSenderAllowed({
      senderId,
      senderName,
      allowFrom: effectiveGroupAllowFrom,
    });
    const commandGate = resolveControlCommandGate({
      useAccessGroups,
      authorizers: [
        { configured: effectiveAllowFrom.length > 0, allowed: senderAllowedForCommands },
        {
          configured: effectiveGroupAllowFrom.length > 0,
          allowed: groupAllowedForCommands,
        },
      ],
      allowTextCommands,
      hasControlCommand,
    });
    const commandAuthorized =
      kind === "direct"
        ? dmPolicy === "open" || senderAllowedForCommands
        : commandGate.commandAuthorized;

    if (kind === "direct") {
      if (dmPolicy === "disabled") {
        logVerboseMessage(`mattermost: drop dm (dmPolicy=disabled sender=${senderId})`);
        return;
      }
      if (dmPolicy !== "open" && !senderAllowedForCommands) {
        if (dmPolicy === "pairing") {
          const { code, created } = await core.channel.pairing.upsertPairingRequest({
            channel: "mattermost",
            id: senderId,
            meta: { name: senderName },
          });
          logVerboseMessage(`mattermost: pairing request sender=${senderId} created=${created}`);
          if (created) {
            try {
              await sendMessageMattermost(
                `user:${senderId}`,
                core.channel.pairing.buildPairingReply({
                  channel: "mattermost",
                  idLine: `Your Mattermost user id: ${senderId}`,
                  code,
                }),
                { accountId: account.accountId },
              );
              opts.statusSink?.({ lastOutboundAt: Date.now() });
            } catch (err) {
              logVerboseMessage(`mattermost: pairing reply failed for ${senderId}: ${String(err)}`);
            }
          }
        } else {
          logVerboseMessage(`mattermost: drop dm sender=${senderId} (dmPolicy=${dmPolicy})`);
        }
        return;
      }
    } else {
      if (groupPolicy === "disabled") {
        logVerboseMessage("mattermost: drop group message (groupPolicy=disabled)");
        return;
      }
      if (groupPolicy === "allowlist") {
        if (effectiveGroupAllowFrom.length === 0) {
          logVerboseMessage("mattermost: drop group message (no group allowlist)");
          return;
        }
        if (!groupAllowedForCommands) {
          // Still record as pending history so other agents' replies
          // appear as conversation context when this agent is @mentioned.
          const earlyTeamId = payload.data?.team_id ?? channelInfo?.team_id ?? undefined;
          const earlyRoute = core.channel.routing.resolveAgentRoute({
            cfg,
            channel: "mattermost",
            accountId: account.accountId,
            teamId: earlyTeamId,
            peer: { kind, id: channelId },
          });
          const earlyThreadRootId = post.root_id?.trim() || undefined;
          const earlyThreadKeys = resolveThreadSessionKeys({
            baseSessionKey: earlyRoute.sessionKey,
            threadId: earlyThreadRootId,
            parentSessionKey: earlyThreadRootId ? earlyRoute.sessionKey : undefined,
          });
          const earlyHistoryKey = earlyThreadKeys.sessionKey;
          const earlyBody =
            rawText ||
            (post.file_ids?.length
              ? `[Mattermost ${post.file_ids.length === 1 ? "file" : "files"}]`
              : "");
          const earlyTrimmed = earlyBody.trim();
          if (earlyHistoryKey && earlyTrimmed) {
            recordPendingHistoryEntryIfEnabled({
              historyMap: channelHistories,
              limit: historyLimit,
              historyKey: earlyHistoryKey,
              entry: {
                sender: senderName,
                body: earlyTrimmed,
                timestamp: typeof post.create_at === "number" ? post.create_at : undefined,
                messageId: post.id ?? undefined,
              },
            });
          }
          logVerboseMessage(
            `mattermost: drop group sender=${senderId} (not in groupAllowFrom, saved as pending history)`,
          );
          return;
        }
      }
    }

    if (kind !== "direct" && commandGate.shouldBlock) {
      logInboundDrop({
        log: logVerboseMessage,
        channel: "mattermost",
        reason: "control command (unauthorized)",
        target: senderId,
      });
      return;
    }

    const teamId = payload.data?.team_id ?? channelInfo?.team_id ?? undefined;
    const channelName = payload.data?.channel_name ?? channelInfo?.name ?? "";
    const channelDisplay =
      payload.data?.channel_display_name ?? channelInfo?.display_name ?? channelName;
    const roomLabel = channelName ? `#${channelName}` : channelDisplay || `#${channelId}`;

    const route = core.channel.routing.resolveAgentRoute({
      cfg,
      channel: "mattermost",
      accountId: account.accountId,
      teamId,
      peer: {
        kind,
        id: kind === "direct" ? senderId : channelId,
      },
    });

    const baseSessionKey = route.sessionKey;
    const threadRootId = post.root_id?.trim() || post.id || undefined;
    const threadKeys = resolveThreadSessionKeys({
      baseSessionKey,
      threadId: threadRootId,
      parentSessionKey: threadRootId ? baseSessionKey : undefined,
    });
    const sessionKey = threadKeys.sessionKey;
    const historyKey = kind === "direct" ? null : sessionKey;

    const mentionRegexes = core.channel.mentions.buildMentionRegexes(cfg, route.agentId);
    const wasMentioned =
      kind !== "direct" &&
      ((botUsername ? rawText.toLowerCase().includes(`@${botUsername.toLowerCase()}`) : false) ||
        core.channel.mentions.matchesMentionPatterns(rawText, mentionRegexes));
    const pendingBody =
      rawText ||
      (post.file_ids?.length
        ? `[Mattermost ${post.file_ids.length === 1 ? "file" : "files"}]`
        : "");
    const pendingSender = senderName;
    const recordPendingHistory = () => {
      const trimmed = pendingBody.trim();
      recordPendingHistoryEntryIfEnabled({
        historyMap: channelHistories,
        limit: historyLimit,
        historyKey: historyKey ?? "",
        entry:
          historyKey && trimmed
            ? {
                sender: pendingSender,
                body: trimmed,
                timestamp: typeof post.create_at === "number" ? post.create_at : undefined,
                messageId: post.id ?? undefined,
              }
            : null,
      });
    };

    const oncharEnabled = account.chatmode === "onchar" && kind !== "direct";
    const oncharPrefixes = oncharEnabled ? resolveOncharPrefixes(account.oncharPrefixes) : [];
    const oncharResult = oncharEnabled
      ? stripOncharPrefix(rawText, oncharPrefixes)
      : { triggered: false, stripped: rawText };
    const oncharTriggered = oncharResult.triggered;

    const shouldRequireMention =
      kind !== "direct" &&
      core.channel.groups.resolveRequireMention({
        cfg,
        channel: "mattermost",
        accountId: account.accountId,
        groupId: channelId,
        requireMentionOverride: account.requireMention,
      });
    const shouldBypassMention =
      isControlCommand && shouldRequireMention && !wasMentioned && commandAuthorized;
    const effectiveWasMentioned = wasMentioned || shouldBypassMention || oncharTriggered;
    const canDetectMention = Boolean(botUsername) || mentionRegexes.length > 0;

    if (oncharEnabled && !oncharTriggered && !wasMentioned && !isControlCommand) {
      recordPendingHistory();
      return;
    }

    if (kind !== "direct" && shouldRequireMention && canDetectMention) {
      if (!effectiveWasMentioned) {
        recordPendingHistory();
        return;
      }
    }
    const mediaList = await resolveMattermostMedia(post.file_ids);
    const imageContents = await buildMattermostImageContents(mediaList, logger);
    const mediaPlaceholder = buildMattermostAttachmentPlaceholder(mediaList);
    const bodySource = oncharTriggered ? oncharResult.stripped : rawText;
    const baseText = [bodySource, mediaPlaceholder].filter(Boolean).join("\n").trim();
    const bodyText = normalizeMention(baseText, botUsername);
    if (!bodyText) {
      return;
    }

    core.channel.activity.record({
      channel: "mattermost",
      accountId: account.accountId,
      direction: "inbound",
    });

    const fromLabel = formatInboundFromLabel({
      isGroup: kind !== "direct",
      groupLabel: channelDisplay || roomLabel,
      groupId: channelId,
      groupFallback: roomLabel || "Channel",
      directLabel: senderName,
      directId: senderId,
    });

    const preview = bodyText.replace(/\s+/g, " ").slice(0, 160);
    const inboundLabel =
      kind === "direct"
        ? `Mattermost DM from ${senderName}`
        : `Mattermost message in ${roomLabel} from ${senderName}`;
    core.system.enqueueSystemEvent(`${inboundLabel}: ${preview}`, {
      sessionKey,
      contextKey: `mattermost:message:${channelId}:${post.id ?? "unknown"}`,
    });

    const textWithId = `${bodyText}\n[mattermost message id: ${post.id ?? "unknown"} channel: ${channelId}]`;
    const body = core.channel.reply.formatInboundEnvelope({
      channel: "Mattermost",
      from: fromLabel,
      timestamp: typeof post.create_at === "number" ? post.create_at : undefined,
      body: textWithId,
      chatType,
      sender: { name: senderName, id: senderId },
    });
    let combinedBody = body;
    if (historyKey) {
      combinedBody = buildPendingHistoryContextFromMap({
        historyMap: channelHistories,
        historyKey,
        limit: historyLimit,
        currentMessage: combinedBody,
        formatEntry: (entry) =>
          core.channel.reply.formatInboundEnvelope({
            channel: "Mattermost",
            from: fromLabel,
            timestamp: entry.timestamp,
            body: `${entry.body}${
              entry.messageId ? ` [id:${entry.messageId} channel:${channelId}]` : ""
            }`,
            chatType,
            senderLabel: entry.sender,
          }),
      });
    }

    const to = kind === "direct" ? `user:${senderId}` : `channel:${channelId}`;
    const mediaPayload = buildAgentMediaPayload(mediaList);
    const inboundHistory =
      historyKey && historyLimit > 0
        ? (channelHistories.get(historyKey) ?? []).map((entry) => ({
            sender: entry.sender,
            body: entry.body,
            timestamp: entry.timestamp,
          }))
        : undefined;
    // Narrow try/catch to context finalization to avoid hiding real bugs in dispatch/delivery
    let ctxPayload: ReturnType<typeof core.channel.reply.finalizeInboundContext>;
    try {
      ctxPayload = core.channel.reply.finalizeInboundContext({
        Body: combinedBody,
        BodyForAgent: bodyText,
        InboundHistory: inboundHistory,
        RawBody: bodyText,
        CommandBody: bodyText,
        From:
          kind === "direct"
            ? `mattermost:${senderId}`
            : kind === "group"
              ? `mattermost:group:${channelId}`
              : `mattermost:channel:${channelId}`,
        To: to,
        SessionKey: sessionKey,
        ParentSessionKey: threadKeys.parentSessionKey,
        AccountId: route.accountId,
        ChatType: chatType,
        ConversationLabel: fromLabel,
        GroupSubject: kind !== "direct" ? channelDisplay || roomLabel : undefined,
        GroupChannel: channelName ? `#${channelName}` : undefined,
        GroupSpace: teamId,
        SenderName: senderName,
        SenderId: senderId,
        Provider: "mattermost" as const,
        Surface: "mattermost" as const,
        MessageSid: post.id ?? undefined,
        MessageSids: allMessageIds.length > 1 ? allMessageIds : undefined,
        MessageSidFirst: allMessageIds.length > 1 ? allMessageIds[0] : undefined,
        MessageSidLast:
          allMessageIds.length > 1 ? allMessageIds[allMessageIds.length - 1] : undefined,
        ReplyToId: threadRootId,
        MessageThreadId: threadRootId,
        Timestamp: typeof post.create_at === "number" ? post.create_at : undefined,
        WasMentioned: kind !== "direct" ? effectiveWasMentioned : undefined,
        CommandAuthorized: commandAuthorized,
        OriginatingChannel: "mattermost" as const,
        OriginatingTo: to,
        ...mediaPayload,
      });
    } catch (err) {
      runtime.error?.(
        `mattermost malformed message (channel: ${channelId}): ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
      logger.debug?.(
        `mattermost error details: ${err instanceof Error && err.stack ? err.stack : String(err)}`,
      );
      // Skip malformed message gracefully to keep monitor running
      return;
    }

    if (kind === "direct") {
      const sessionCfg = cfg.session;
      const storePath = core.channel.session.resolveStorePath(sessionCfg?.store, {
        agentId: route.agentId,
      });
      await core.channel.session.updateLastRoute({
        storePath,
        sessionKey: route.mainSessionKey,
        deliveryContext: {
          channel: "mattermost",
          to,
          accountId: route.accountId,
        },
      });
    }

    const previewLine = bodyText.slice(0, 200).replace(/\n/g, "\\n");
    logVerboseMessage(
      `mattermost inbound: from=${ctxPayload.From} len=${bodyText.length} preview="${previewLine}"`,
    );

    const textLimit = core.channel.text.resolveTextChunkLimit(
      cfg,
      "mattermost",
      account.accountId,
      {
        fallbackLimit: account.textChunkLimit ?? 4000,
      },
    );
    const tableMode = core.channel.text.resolveMarkdownTableMode({
      cfg,
      channel: "mattermost",
      accountId: account.accountId,
    });

    const { onModelSelected, ...prefixOptions } = createReplyPrefixOptions({
      cfg,
      agentId: route.agentId,
      channel: "mattermost",
      accountId: account.accountId,
    });

    const typingCallbacks = createTypingCallbacks({
      start: () => sendTypingIndicator(channelId, threadRootId),
      onStartError: (err) => {
        logTypingFailure({
          log: (message) => logger.debug?.(message),
          channel: "mattermost",
          target: channelId,
          error: err,
        });
      },
    });
    let markDispatchIdle: (() => void) | undefined;
    try {
      const {
        dispatcher,
        replyOptions,
        markDispatchIdle: markIdle,
      } = core.channel.reply.createReplyDispatcherWithTyping({
        ...prefixOptions,
        humanDelay: core.channel.reply.resolveHumanDelayConfig(cfg, route.agentId),
        deliver: async (payload: ReplyPayload) => {
          const mediaUrls = payload.mediaUrls ?? (payload.mediaUrl ? [payload.mediaUrl] : []);
          const text = core.channel.text.convertMarkdownTables(payload.text ?? "", tableMode);
          if (mediaUrls.length === 0) {
            const chunkMode = core.channel.text.resolveChunkMode(
              cfg,
              "mattermost",
              account.accountId,
            );
            const chunks = core.channel.text.chunkMarkdownTextWithMode(text, textLimit, chunkMode);
            for (const chunk of chunks.length > 0 ? chunks : [text]) {
              if (!chunk) {
                continue;
              }
              await sendMessageMattermost(to, chunk, {
                accountId: account.accountId,
                replyToId: threadRootId,
              });
            }
          } else {
            let first = true;
            for (const mediaUrl of mediaUrls) {
              const caption = first ? text : "";
              first = false;
              await sendMessageMattermost(to, caption, {
                accountId: account.accountId,
                mediaUrl,
                replyToId: threadRootId,
              });
            }
          }
          runtime.log?.(`delivered reply to ${to}`);
        },
        onError: (err, info) => {
          runtime.error?.(`mattermost ${info.kind} reply failed: ${String(err)}`);
        },
        onReplyStart: typingCallbacks.onReplyStart,
      });
      markDispatchIdle = markIdle;

      await core.channel.reply.dispatchReplyFromConfig({
        ctx: ctxPayload,
        cfg,
        dispatcher,
        replyOptions: {
          ...replyOptions,
          images: imageContents.length > 0 ? imageContents : undefined,
          disableBlockStreaming:
            typeof account.blockStreaming === "boolean" ? !account.blockStreaming : undefined,
          onModelSelected,
        },
      });

      if (historyKey) {
        clearHistoryEntriesIfEnabled({
          historyMap: channelHistories,
          historyKey,
          limit: historyLimit,
        });
      }
    } finally {
      // Always mark dispatcher idle to prevent stuck typing indicators
      markDispatchIdle?.();
    }
  };

  const inboundDebounceMs = core.channel.debounce.resolveInboundDebounceMs({
    cfg,
    channel: "mattermost",
  });
  const debouncer = core.channel.debounce.createInboundDebouncer<{
    post: MattermostPost;
    payload: MattermostEventPayload;
  }>({
    debounceMs: inboundDebounceMs,
    buildKey: (entry) => {
      const channelId =
        entry.post.channel_id ??
        entry.payload.data?.channel_id ??
        entry.payload.broadcast?.channel_id;
      if (!channelId) {
        return null;
      }
      const channelType = entry.payload.data?.channel_type;
      const kind = channelKind(channelType);
      const threadId =
        entry.post.root_id?.trim() || entry.post.id || undefined;
      const threadKey = threadId ? `thread:${threadId}` : "channel";
      return `mattermost:${account.accountId}:${channelId}:${threadKey}`;
    },
    shouldDebounce: (entry) => {
      if (entry.post.file_ids && entry.post.file_ids.length > 0) {
        return false;
      }
      const text = entry.post.message?.trim() ?? "";
      if (!text) {
        return false;
      }
      return !core.channel.text.hasControlCommand(text, cfg);
    },
    onFlush: async (entries) => {
      const last = entries.at(-1);
      if (!last) {
        return;
      }
      if (entries.length === 1) {
        await handlePost(last.post, last.payload);
        return;
      }
      const combinedText = entries
        .map((entry) => entry.post.message?.trim() ?? "")
        .filter(Boolean)
        .join("\n");
      const mergedPost: MattermostPost = {
        ...last.post,
        message: combinedText,
        file_ids: [],
      };
      const ids = entries.map((entry) => entry.post.id).filter(Boolean);
      await handlePost(mergedPost, last.payload, ids.length > 0 ? ids : undefined);
    },
    onError: (err) => {
      runtime.error?.(`mattermost debounce flush failed: ${String(err)}`);
    },
  });

  const wsUrl = buildMattermostWsUrl(baseUrl);
  let seq = 1;
  const connectOnce = createMattermostConnectOnce({
    wsUrl,
    botToken,
    abortSignal: opts.abortSignal,
    statusSink: opts.statusSink,
    runtime,
    webSocketFactory: opts.webSocketFactory,
    nextSeq: () => seq++,
    onPosted: async (post, payload) => {
      await debouncer.enqueue({ post, payload });
    },
  });

  await runWithReconnect(connectOnce, {
    abortSignal: opts.abortSignal,
    jitterRatio: 0.2,
    onError: (err) => {
      runtime.error?.(`mattermost connection failed: ${String(err)}`);
      opts.statusSink?.({ lastError: String(err), connected: false });
    },
    onReconnect: (delayMs) => {
      runtime.log?.(`mattermost reconnecting in ${Math.round(delayMs / 1000)}s`);
    },
  });
}
