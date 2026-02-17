# @openclaw/mattermost

OpenClaw Mattermost channel plugin — self-hosted Slack-style chat integration.

## Setup

```bash
npm install
```

## Development

```bash
npm run typecheck   # Type-check without emitting
npm run test        # Run tests
```

## Usage

This plugin is loaded by OpenClaw via the `plugins.load.paths` config. Point it at this directory:

```json
{
  "plugins": {
    "load": {
      "paths": ["/path/to/openclaw-mattermost"]
    }
  }
}
```
