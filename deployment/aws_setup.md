# AWS EC2 Node Setup (Node 1 — "credit-card transactions")

**Role in FedGuard:** runs `fl_client.py --partition node1`. It trains on its own
copy of `data/node1_partition.csv` and connects **outbound** to the Flower
orchestrator. It never receives anyone else's data.

**Actual instance used in this project**

| Field | Value |
|---|---|
| Instance ID | `i-0df46123c3ab0b57c` |
| Type | `t3.micro` (free tier: 2 vCPU, 1 GB) |
| AMI | Ubuntu 24.04 LTS (`ami-006f82a1d5a27da54`) |
| Region / AZ | `ap-south-1` / `ap-south-1b` |
| Public IP | `65.1.248.113` (recorded in `.env` as `AWS_EC2_PUBLIC_IP`) |
| Key pair | `MinorProject` → local `~/.ssh/MinorProject.pem` |
| Security group | `sg-071213d6eb2c89c40` |

> The project's **orchestrator runs on the GCP VM** (see `gcp_setup.md`), so this
> machine needs **no inbound port 8080** — clients dial out to the server.

---

## 1. Launch the instance (skip if it already exists)

```bash
aws ec2 run-instances \
  --image-id ami-006f82a1d5a27da54 \
  --instance-type t3.micro \
  --key-name MinorProject \
  --security-group-ids sg-071213d6eb2c89c40 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fedguard-node1}]' \
  --query 'Instances[0].InstanceId' --output text
```

Record the public IP in `.env`:

```bash
aws ec2 describe-instances --instance-ids <ID> \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

## 2. Security group rules (security best practice)

```bash
# SSH: only YOUR current public IP, never 0.0.0.0/0.
MY_IP=$(curl -s https://api.ipify.org)
aws ec2 authorize-security-group-ingress --group-id sg-071213d6eb2c89c40 \
  --protocol tcp --port 22 --cidr "$MY_IP/32"
```

*Why not `0.0.0.0/0`:* an internet-wide SSH port on a public cloud VM is
scanned within minutes of launch. Restricting source CIDR to one `/32` is the
minimum bar for a demo account.
*Dynamic home IPs* will change (this project had to re-authorise SSH after the
laptop moved from mobile hotspot to Wi-Fi) — re-run the command above whenever
SSH times out.

No inbound rule for 8080 is required: the node only initiates outbound
connections to the orchestrator.

## 3. Connect and install dependencies

```bash
ssh -i ~/.ssh/MinorProject.pem ubuntu@65.1.248.113

sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ~200 MB CPU wheel
pip install -r requirements.txt
```

## 4. Copy the code and this node's data

From your laptop (never copy the whole `data/raw/` dataset, and never copy
`.env` — this VM needs no credentials):

```bash
KEY=~/.ssh/MinorProject.pem HOST=65.1.248.113
scp -i $KEY src/fl_client.py src/model.py src/data_utils.py \
             src/dp_engine.py src/metrics_logger.py requirements.txt \
             ubuntu@$HOST:~
scp -i $KEY deployment/run_node.sh ubuntu@$HOST:~
scp -i $KEY data/node1_partition.csv ubuntu@$HOST:data_partition_tmp.csv
ssh -i $KEY ubuntu@$HOST 'mkdir -p data && mv data_partition_tmp.csv data/node1_partition.csv && chmod +x run_node.sh'
```

`fl_client.py` is deliberately cloud-agnostic — the identical file is also
running on the GCP VM; only `--partition` and `--server_address` differ.

## 5. Start this node

```bash
./run_node.sh node1 8.231.92.55 8080            # orchestrator = GCP VM
./run_node.sh node1 8.231.92.55 8080 --use-dp    # with differential privacy
```

## 6. Validate

```bash
tail -f nohup.out   # expect: "[client:node1] fit() sent 4097 weight values (0 raw rows)"
```

- [ ] The log shows `sent 4097 weight values (0 raw rows)` each round — proof
      only weights, never raw transactions, cross the network.
- [ ] The orchestrator (GCP side) reports 2 connected clients with this node's
      public IP visible in its connection log.

## Costs

`t3.micro` on the free tier = $0 for 750 hours/month. **Stop the instance when
not demoing** so it doesn't fall outside the free allowance:

```bash
aws ec2 stop-instances --instance-ids i-0df46123c3ab0b57c
```
