# optionscanner

Optionscanner is a lightweight toolkit for monitoring option chains and portfolio exposure using the Interactive Brokers (IBKR) API.

## Docker usage

The repository ships with a `docker-compose.yml` file that starts both the IBKR Gateway and the scanner in a shared network. Follow the steps below to bring the stack online:

1. **Create credentials file (optional but recommended).**
   Create a `.env` file in the project root so Docker Compose can read your credentials and gateway preferences. At minimum, populate the following values:

   ```dotenv
   TWS_USERID=your-ibkr-username
   TWS_PASSWORD=your-ibkr-password
   IAPI_CLIENT_ID=1
   TRADING_MODE=paper
   ```

   The variables are forwarded directly to the `ghcr.io/ibkrcampus/ibkr-gateway` image. Refer to the image documentation for the full list of supported options (for example, 2FA timeout or region-specific settings).

2. **Prepare persistent storage for the gateway.**
   Create directories that will be mounted into the gateway container so that session settings and logs survive container restarts:

   ```bash
   mkdir -p gateway/config gateway/logs
   ```

   The compose file mounts these folders at `/home/ibkr/.ibgateway` and `/home/ibkr/logs`, respectively. They are excluded from version control via `gateway/.gitignore` so they can safely hold runtime data.

3. **Review application configuration.**
   The default `config.yaml` now targets the gateway container hostname:

   ```yaml
   ibkr:
     host: "ibkr-gateway"
   ```

   If you run the scanner outside of Docker or connect to a remote gateway, update `ibkr.host` to the appropriate hostname or IP address.

4. **Start the stack.**
   Build and launch both services in the foreground:

   ```bash
   docker compose up --build
   ```

   Docker Compose creates a user-defined bridge network named `ibkr-net` so the scanner can reach the gateway at `ibkr-gateway:4002`. Gateway ports `4001`/`4002` are published to the host for API access, and `5900` exposes the VNC session required to complete IBKR logins and 2FA from a VNC client. Logs from both containers appear in the same terminal. Use `Ctrl+C` to stop the stack.

5. **Complete the IBKR login.**
   Connect a VNC client to `localhost:5900` (password is provided by the gateway image documentation) to approve the interactive login or supply 2FA codes when prompted. The session only needs to remain open long enough for the gateway to finish initialization.

6. **Run in the background (optional).**
   To run detached, use `docker compose up --build -d`. Tail the combined logs with `docker compose logs -f`.

7. **Clean up.**
   Stop the services and remove the containers with `docker compose down`. Persistent gateway data remains inside the `gateway/` directory.

## Local development

When running the scanner directly on your workstation, ensure the IBKR Gateway is reachable and adjust `config.yaml` accordingly. Install dependencies with `pip install -r requirements.txt` and run `python main.py`.
