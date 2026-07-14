# Deploying NeuroBlock

NeuroBlock ships as **one container**: Flask serves the API, the WebSocket
training stream, *and* the built React app from a single origin. No CORS, no
URL wiring, one thing to run.

```
docker build -t neuroblock .
docker run -p 8080:8080 -e SECRET_KEY="$(openssl rand -base64 32)" neuroblock
# → http://localhost:8080 is the whole app
```

Everything below is just "where does that container run."

---

## Option A — AWS EC2 (recommended for the Turing summer program)

One VM, started for class weeks, stopped the rest of the year. No cold
starts during lectures, predictable cost, and trivially debuggable.

**Sizing:** `t3.large` (2 vCPU / 8 GB) handles a classroom on small datasets.
Use `c7i.xlarge` (4 vCPU) if many students train MNIST/CIFAR simultaneously.
Cost while running: ~$0.08–0.17/hr. Stopped: pennies (disk only).

### 1. Launch the instance (once)

- AMI: Ubuntu 24.04 LTS, instance type `t3.large`, 30 GB disk
- Security group: allow inbound **22** (your IP) and **80/443** (anywhere)
- Allocate an **Elastic IP** and associate it (keeps the address across stops)

### 2. Install Docker and the app (once)

```bash
ssh ubuntu@<ELASTIC_IP>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && exit   # re-ssh after this

ssh ubuntu@<ELASTIC_IP>
git clone https://github.com/aryamangoenka/DND-Neural-Network.git
cd DND-Neural-Network
echo "SECRET_KEY=$(openssl rand -base64 32)" > .env
docker compose up -d --build     # first build ~5-10 min
```

App is now at `http://<ELASTIC_IP>:8080`.

### 3. HTTPS on port 443 (recommended, ~2 min)

Point a domain/subdomain (e.g. `neuroblock.yourdomain.org`) at the Elastic
IP, then run Caddy — it fetches TLS certificates automatically:

```bash
docker run -d --name caddy --network host \
  -v caddy_data:/data caddy:2 \
  caddy reverse-proxy --from neuroblock.yourdomain.org --to localhost:8080
```

(No domain? The app works fine over `http://<ELASTIC_IP>:8080` for a
classroom.)

### 4. Operating it

```bash
# before class weeks
aws ec2 start-instances --instance-ids i-xxxx     # or EC2 console → Start

# after the program
aws ec2 stop-instances --instance-ids i-xxxx      # billing stops, disk kept

# update to latest code
ssh ubuntu@<ELASTIC_IP> "cd DND-Neural-Network && git pull && docker compose up -d --build"

# logs
ssh ubuntu@<ELASTIC_IP> "cd DND-Neural-Network && docker compose logs -f --tail 100"
```

---

## Option B — Google Cloud Run (scale-to-zero fallback)

Costs ~$0 when idle, but cold-starts ~20–30 s after inactivity (TensorFlow
loading). Requires a GCP project with billing.

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
./deploy-gcp.sh          # enables APIs, creates secret + registry, builds, deploys
```

Redeploys after the first run: `gcloud builds submit --config cloudbuild.yaml`

Tip for class weeks: `gcloud run services update neuroblock --min-instances=1`
(removes cold starts, ~$25/mo while set; set back to 0 afterwards).

---

## Architecture constraints (why max one instance)

The backend keeps training state in-process and on local disk (saved model
architecture, per-session datasets, stop flags). Run **exactly one
instance / one gunicorn worker** — this is already encoded in the
Dockerfile CMD and `cloudbuild.yaml` (`--max-instances=1`). For a classroom
(~30 students, small datasets) this is fine; concurrent trainings share the
CPU and simply take a bit longer.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `PORT` | 8080 | Listen port |
| `FLASK_CONFIG` | production (in container) | `development` / `production` |
| `SECRET_KEY` | — | Session signing key. Set a real one. |
| `EXTRA_ALLOWED_ORIGINS` | empty | Comma-separated extra CORS origins (only needed for split deployments) |
| `SESSION_COOKIE_SECURE` | true | Set `false` when serving plain HTTP (bare-IP EC2) |
| `SESSION_MAX_AGE` | 168 | Hours before per-session datasets are cleaned |
