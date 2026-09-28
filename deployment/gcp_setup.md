# GCP VM Setup (Orchestrator + Node 2)

**Role in FedGuard:** runs **both** the Flower orchestrator
(`fl_server.py`) and `fl_client.py --partition node2`. The AWS node dials into
this machine over the public internet.

> **Why the orchestrator lives here:** the developer laptop sits behind carrier
> CGNAT (no inbound connections possible), so option "laptop + ngrok" would have
> needed a paid tunnel account. Hosting the server on this VM gives a stable
> public IP, no tunnel, no extra account. Each party's raw CSV still never
> leaves its own cloud — only model weights cross the GCP↔AWS boundary.

**Actual VM used in this project**

| Field | Value |
|---|---|
| VM name | `minorproject` |
| Machine type | `e2-micro` (free tier: 1 shared vCPU, 1 GB) |
| Zone | `asia-south1-c` |
| Project | `minor-project-509809` |
| Public IP | `8.231.92.55` (in `.env` as `GCP_VM_PUBLIC_IP`) |
| Login | `gcloud compute ssh` / key `~/.ssh/google_compute_engine` |
| OS | Debian/Ubuntu gcloud image, Python 3.13 |

---

## 1. Create the VM (skip if it already exists)

```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

# One-time: free-tier VMs are only allowed in specific zones; enable the API first
gcloud services enable compute.googleapis.com --project=minor-project-509809

gcloud compute instances create minorproject \
  --project=minor-project-509809 \
  --zone=asia-south1-c \
  --machine-type=e2-micro \
  --image-family=debian-12 --image-project=debian-cloud \
  --tags=fedguard-orchestrator
```

Record the IP in `.env`: `GCP_VM_PUBLIC_IP=<external ip>`.

## 2. Firewall — allow Flower, but restricted (security best practice)

The orchestrator listens on `0.0.0.0:8080`, so an inbound rule is required —
restricted to the AWS node's IP, **not** `0.0.0.0/0`:

```bash
AWS_NODE_IP=65.1.248.113

# remove the wide-open rule if present
gcloud compute firewall-rules delete allow-flower-8080 --quiet \
  --project=minor-project-509809

gcloud compute firewall-rules create allow-flower-8080 \
  --project=minor-project-509809 \
  --network=default \
  --direction=INGRESS --action=ALLOW --rules=tcp:8080 \
  --source-ranges="$AWS_NODE_IP/32" \
  --target-tags=fedguard-orchestrator
```

```bash
# SSH: default-allow-ssh (0.0.0.0/0) ships with the default network; for the
# demo tighten it the same way as AWS if your account policy requires it.
gcloud compute firewall-rules describe default-allow-ssh \
  --project=minor-project-509809
```

## 3. Install dependencies

```bash
gcloud compute ssh minorproject --zone=asia-south1-c --project=minor-project-509809

sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 4. Copy code + this node's data

From the laptop (again: **no `.env`**, no raw Kaggle dataset):

```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
gcloud compute scp --project=minor-project-509809 --zone=asia-south1-c \
  src/fl_server.py src/fl_client.py src/model.py src/data_utils.py \
  src/dp_engine.py src/metrics_logger.py requirements.txt \
  minorproject:~/

gcloud compute scp --project=minor-project-509809 --zone=asia-south1-c \
  data/node2_partition.csv minorproject:~/node2_partition_tmp.csv

gcloud compute ssh minorproject --zone=asia-south1-c --project=minor-project-509809 \
  --command='mkdir -p data && mv node2_partition_tmp.csv data/node2_partition.csv'
```

## 5. Run — orchestrator + co-located client

```bash
# terminal 1 (VM): the orchestrator
.venv/bin/python fl_server.py --num_rounds 5 \
  --server_address 0.0.0.0:8080 \
  --source federated_cloud --append

# terminal 2 (VM): node2's training client
./run_node.sh node2 localhost 8080

# terminal 3 (laptop): the cross-cloud client
ssh -i ~/.ssh/MinorProject.pem ubuntu@65.1.248.113
./run_node.sh node1 8.231.92.55 8080
```

For the privacy-preserving variant add `--use-dp` to **both** client commands
(and start with `--source federated_dp --append`).

## 6. Validate a real cross-cloud round

```bash
# on AWS VM — must show only outbound connection to the GCP IP
curl -s -o /dev/null -w '%{http_code}\n' --max-time 5 http://8.231.92.55:8080   # timeout/000 is normal (gRPC)

# on GCP VM — orchestrator log must show 2 connected clients:
tail -f server.log
#   [server] round 1: 2 clients, metrics {accuracy:..., recall:...}
```

- [ ] 2 clients connected (AWS public IP + localhost)
- [ ] `results/cloud_server.log` captured on the orchestrator (includes the
      `client connected ... peer=ipv4:<AWS public IP>` lines)
- [ ] Every client log line reads `sent 4097 weight values (0 raw rows)`
- [ ] ε increases monotonically when running with `--use-dp`

## Costs

`e2-micro` in `asia-south1-c` is inside GCP's always-free allowance
(≈24×7 in `us-west1`/`us-central1`/`us-east1`; other zones limited) — stop the
VM after the demo if it bills in your region:

```bash
gcloud compute instances stop minorproject --zone=asia-south1-c --project=minor-project-509809
```
