# GitHub Actions Deployment with Tailscale

This workflow deploys the optionscanner to a Raspberry Pi using Tailscale for secure connectivity.

## Required GitHub Secrets

Configure these secrets in your GitHub repository settings (Settings → Secrets and variables → Actions):

### Tailscale Secrets

| Secret | Description | Example |
|--------|-------------|---------|
| `TAILSCALE_AUTH_KEY` | Tailscale auth key for CI runner authentication | `tskey-auth-...` |
| `TAILSCALE_IP` | Tailscale IP address of the Raspberry Pi | `100.x.y.z` |
| `TAILSCALE_SSH_USER` | SSH username on the Raspberry Pi | `pi` |
| `TAILSCALE_SSH_KEY` | SSH private key (ed25519 recommended) for authentication | `-----BEGIN OPENSSH PRIVATE KEY-----...` |

### Application Secrets

| Secret | Description | Required |
|--------|-------------|----------|
| `TWS_USERID` | IBKR username | Yes |
| `TWS_PASSWORD` | IBKR password | Yes |
| `IAPI_CLIENT_ID` | IBKR client ID | Yes (default: 1) |
| `TRADING_MODE` | `paper` or `live` | Yes |
| `VNC_SERVER_PASSWORD` | VNC access password | Optional |
| `READ_ONLY_API` | `yes` or `no` | Yes |
| `SLACK_WEBHOOK_URL` | Slack webhook for notifications | Yes |
| `SLACK_TEST_CHANNEL` | Slack channel for testing | Optional |
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `IBKR_MARKET_DATA_TYPE` | Market data type (1=live, 2=frozen, 3=delayed) | Yes |
| `DISABLE_PORTFOLIO_MANAGER` | `yes` or `no` | Yes |
| `LOKI_URL` | Grafana Loki URL | Optional |
| `LOKI_USERNAME` | Grafana Loki username | Optional |
| `LOKI_PASSWORD` | Grafana Loki password | Optional |
| `REDDIT_CLIENT_ID` | Reddit API client ID (for Whale Following) | Optional |
| `REDDIT_CLIENT_SECRET` | Reddit API client secret (for Whale Following) | Optional |
| `REDDIT_USER_AGENT` | Reddit API user agent | Optional |

## Tailscale Setup

### 1. Generate an Auth Key

1. Go to your Tailscale admin console: https://login.tailscale.com/admin/settings/keys
2. Click "Generate auth key"
3. Create a key with:
   - **Reusable**: Yes (for CI/CD)
   - **Ephemeral**: No
   - **Tags**: `tag:ci` (for ACL purposes)
4. Copy the key and add it as `TAILSCALE_AUTH_KEY` secret

### 2. Find Your Pi's Tailscale IP

```bash
# On your Raspberry Pi
tailscale ip
```

Add this IP as the `TAILSCALE_IP` secret.

### 3. Enable SSH on the Pi

Ensure SSH is enabled and accessible via Tailscale:

```bash
# Check SSH is running
sudo systemctl status ssh

# Enable if needed
sudo systemctl enable ssh
sudo systemctl start ssh
```

### 4. Generate SSH Key for GitHub Actions

```bash
# Generate ed25519 key (do this on your local machine)
ssh-keygen -t ed25519 -f tailscale-deploy-key -C "github-actions-deploy"

# Copy public key to Pi
ssh-copy-id -i tailscale-deploy-key.pub pi@<tailscale-ip>

# Add private key as GitHub secret
cat tailscale-deploy-key | gh secret set TAILSCALE_SSH_KEY
```

## How It Works

1. **Checkout**: Code is checked out from GitHub
2. **Tailscale Connect**: Runner authenticates to your Tailscale network
3. **SSH Setup**: SSH key is configured for the runner
4. **Code Transfer**: Files are synced via rsync to the Pi (excluding `.git`, `.venv`, etc.)
5. **Remote Deploy**: SSH into Pi and:
   - Backup existing config
   - Move new files into place
   - Write `.env` from secrets
   - Pull latest ib-gateway image
   - Rebuild and restart app container
6. **Verification**: Display container status and logs

**Note**: Code is deployed to `~/optionscanner` (home directory) on the Raspberry Pi.

## Triggering Deployment

### Automatic
Pushes to `main` branch trigger automatic deployment.

### Manual
Go to Actions → "Deploy to Raspberry Pi" → "Run workflow" to manually trigger.

## Troubleshooting

### Tailscale Connection Fails
- Verify auth key is valid and not expired
- Check Pi is connected to Tailscale: `tailscale status`
- Ensure firewall allows Tailscale traffic

### SSH Connection Fails
- Verify SSH is running on Pi: `sudo systemctl status ssh`
- Check SSH key is properly installed: `ssh -i ~/.ssh/id_ed25519 pi@<ip>`
- Verify `TAILSCALE_IP` and `TAILSCALE_SSH_USER` secrets are correct

### Docker Compose Fails
- SSH into Pi and check: `docker compose ps`
- View logs: `docker logs algo-trader-app`
- Check disk space: `df -h`

## Security Considerations

1. **Auth Key Protection**: The Tailscale auth key grants network access - treat it as sensitive
2. **SSH Key Security**: Use a dedicated deploy key, not your personal SSH key
3. **Tailscale ACLs**: Consider restricting `tag:ci` permissions in your ACL policy
4. **Secret Rotation**: Rotate all secrets periodically via GitHub and Tailscale consoles
