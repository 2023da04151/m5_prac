# Complete MLflow on AWS Setup Runbook

## EC2 + RDS PostgreSQL + S3 + MLflow Basic Authentication

This runbook consolidates the recorded setup, commands, configuration
files, AWS console settings to verify, troubleshooting, and the next
local-client test. It separates confirmed facts from values that must be
checked in the AWS console or on the instance.

> **Credential safety:** Passwords, database credentials, private keys,
> and Flask secrets are intentionally placeholders. Never commit them to
> Git or paste them into shared logs. Do not make RDS publicly
> accessible merely to simplify connectivity.

------------------------------------------------------------------------

## 1. Architecture and data flow

``` text
Local Python / MLflow client
        |
        | HTTP + Basic Auth (direct port 5000 or SSH tunnel)
        v
EC2 instance — Amazon Linux 2023
MLflow 3.1.4, systemd service, port 5000
        |                                  |
        | PostgreSQL/TLS, port 5432        | AWS SDK / HTTPS, port 443
        v                                  v
RDS PostgreSQL                            Amazon S3
mlflowdb                                  amrith-mlflow-artifacts/
tracking metadata                         mlflow-artifacts/
                                          model/data/artifact files

Separate RDS database: mlflow_authdb
Basic Auth users and permissions
```

### What goes where?

-   **RDS database `mlflowdb`:** experiments, runs, parameters, metrics,
    tags, model registry metadata, and other MLflow metadata.
-   **RDS database `mlflow_authdb`:** MLflow Basic Auth user and
    permission information.
-   **S3 bucket/prefix:** files logged as artifacts, such as model
    files, plots, reports, and test files.
-   **EC2:** runs the MLflow tracking server and accesses S3 using its
    attached IAM role.

------------------------------------------------------------------------

## 2. Confirmed resource inventory

  ---------------------------------------------------------------------------------------------------
  Item                                Recorded value
  ----------------------------------- ---------------------------------------------------------------
  AWS Region                          `us-east-1`

  EC2 instance ID                     `i-06d13b058fbd07a49`

  Operating system                    Amazon Linux 2023

  EC2 login user                      `ec2-user`

  IAM role observed on EC2            `MLflowArtifactBucketAccess`

  MLflow version                      `3.1.4`

  MLflow executable                   `/home/ec2-user/.local/bin/mlflow`

  Python package path observed        `/home/ec2-user/.local/lib/python3.9/site-packages`

  RDS endpoint                        `mlflow-metadata-db.ckh0wmiwcs1t.us-east-1.rds.amazonaws.com`

  RDS port                            `5432`

  Tracking database                   `mlflowdb`

  Database user                       `mlflowadmin`

  Authentication database             `mlflow_authdb`

  S3 bucket                           `amrith-mlflow-artifacts`

  S3 prefix                           `mlflow-artifacts/`

  MLflow server port                  `5000`

  Basic Auth administrator username   `admin`

  Test regular username               `viewer_test`

  Final recorded default permission   `MANAGE`
  ---------------------------------------------------------------------------------------------------

The exact current EC2 public IP/DNS, VPC ID, subnet IDs, security-group
IDs, RDS engine minor version, and all S3 bucket properties were not
preserved in the available setup record. Check the AWS console for those
values; do not guess them.

------------------------------------------------------------------------

## 3. AWS console setup and settings

### 3.1 EC2 instance

In **AWS Console → EC2 → Instances**, locate instance
`i-06d13b058fbd07a49`.

Verify:

1.  Region is `us-east-1`.
2.  Instance state is `Running`.
3.  Operating system is Amazon Linux 2023.
4.  IAM role attached to the instance is `MLflowArtifactBucketAccess`.
5.  Record the current public IPv4 address or public DNS if using direct
    public connectivity or SSH.
6.  Under **Security**, identify the security group attached to the
    instance.
7.  Confirm the instance can reach RDS and S3.

Connect using SSH if you have a private key and the instance has a
reachable address:

``` bash
ssh -i /path/to/your-key.pem ec2-user@<EC2_PUBLIC_DNS>
```

Replace the key path and host with the actual values. Restrict
private-key permissions on Linux/WSL if needed:

``` bash
chmod 400 /path/to/your-key.pem
```

Once connected:

``` bash
whoami
cat /etc/os-release
```

Expected login user: `ec2-user`.

### 3.2 EC2 security group

In **EC2 → Instances → select instance → Security → Security groups**,
inspect inbound and outbound rules.

For SSH access: - Type: SSH - Protocol: TCP - Port: `22` - Source: your
trusted public IP with `/32`

For direct MLflow access: - Type: Custom TCP - Protocol: TCP - Port:
`5000` - Source: your trusted public IP with `/32`

Do **not** open port `5000` to `0.0.0.0/0` just to make the client work.
An SSH tunnel avoids exposing port `5000` publicly.

Outbound connectivity must allow the instance to reach: - RDS on TCP
`5432` - S3 over HTTPS/TCP `443` (through the internet/NAT path or an
appropriate S3 VPC endpoint) - Any required package repositories if
installing packages

The exact existing security-group rules are not present in the record.
Verify them instead of assuming they were already configured.

### 3.3 RDS PostgreSQL

In **AWS Console → RDS → Databases**, open the database associated with:

``` text
mlflow-metadata-db.ckh0wmiwcs1t.us-east-1.rds.amazonaws.com
```

Verify: - Engine: PostgreSQL - Port: `5432` - Database connectivity from
EC2 - RDS security group inbound rule permits TCP `5432` from the EC2
security group - RDS is not unnecessarily public - TLS is used by the
client URI (`sslmode=require`)

Prefer a security-group reference to the EC2 security group rather than
allowing all IPs. The RDS DB identifier, subnet group, VPC ID, exact
security-group ID, and public-accessibility setting were not preserved.

### 3.4 S3 bucket

The configured bucket/prefix is:

``` text
s3://amrith-mlflow-artifacts/mlflow-artifacts/
```

In **S3 → Buckets → `amrith-mlflow-artifacts`**, verify: - Bucket region
is `us-east-1` - Block Public Access remains enabled - The EC2 IAM role
has the necessary object permissions - The bucket is not relying on
public write access

The exact encryption, versioning, lifecycle, and bucket-policy settings
were not recorded. Inspect those in the console rather than assuming a
specific value.

### 3.5 IAM role and policy

In **IAM → Roles → `MLflowArtifactBucketAccess`**, verify: - EC2 is
trusted to assume the role - The role is attached to the EC2 instance -
The policy permits listing the bucket and operating on objects under the
artifact prefix

The required permission pattern is approximately:

``` json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::amrith-mlflow-artifacts"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::amrith-mlflow-artifacts/mlflow-artifacts/*"
    }
  ]
}
```

This is a **policy pattern for verification**, not proof that the
currently attached policy has these exact statements. Keep only
permissions required by your workflow. Do not store long-lived AWS
access keys on EC2 when an instance role can be used.

------------------------------------------------------------------------

## 4. Verify EC2 identity and S3 access

Run these commands on EC2.

### 4.1 Confirm the AWS identity

``` bash
aws sts get-caller-identity
```

This confirms which AWS identity the CLI is using. During setup, the EC2
role was successfully used.

### 4.2 List the artifact prefix

``` bash
aws s3 ls s3://amrith-mlflow-artifacts/mlflow-artifacts/
```

The recorded listing included `mlflow-test.txt`.

### 4.3 Upload a test file

``` bash
echo "MLflow S3 connectivity test" > /tmp/mlflow-test.txt

aws s3 cp \
  /tmp/mlflow-test.txt \
  s3://amrith-mlflow-artifacts/mlflow-artifacts/mlflow-test.txt
```

Verify:

``` bash
aws s3 ls s3://amrith-mlflow-artifacts/mlflow-artifacts/
```

Optional download check:

``` bash
aws s3 cp \
  s3://amrith-mlflow-artifacts/mlflow-artifacts/mlflow-test.txt \
  /tmp/mlflow-test-downloaded.txt

cat /tmp/mlflow-test-downloaded.txt
```

This verifies EC2-to-S3 access. It does not alone prove that a model
artifact can be uploaded by the MLflow client; perform the end-to-end
run test later.

------------------------------------------------------------------------

## 5. Verify MLflow and Python on EC2

The recorded executable is:

``` text
/home/ec2-user/.local/bin/mlflow
```

Check the version:

``` bash
/home/ec2-user/.local/bin/mlflow --version
```

Expected version: `3.1.4`.

Check Python:

``` bash
python3 --version
```

Inspect package installation:

``` bash
python3 -m pip show mlflow
```

The observed package location was:

``` text
/home/ec2-user/.local/lib/python3.9/site-packages
```

The original installation command was not preserved, so this runbook
does not claim an exact historical `pip install` command. For a fresh
installation, use a controlled Python environment and install the
intended pinned version.

------------------------------------------------------------------------

## 6. RDS database configuration

### 6.1 Main metadata database URI

The environment variable used for the tracking backend was
`MLFLOW_BACKEND_STORE_URI`. The URI format was:

``` text
postgresql://mlflowadmin:<DB_PASSWORD>@mlflow-metadata-db.ckh0wmiwcs1t.us-east-1.rds.amazonaws.com:5432/mlflowdb?sslmode=require
```

The password is a placeholder. Do not paste the real value into logs,
shell history, or Git.

### 6.2 Connect with `psql`

From a host that can reach RDS and has the PostgreSQL client installed:

``` bash
psql "host=mlflow-metadata-db.ckh0wmiwcs1t.us-east-1.rds.amazonaws.com port=5432 dbname=mlflowdb user=mlflowadmin sslmode=require"
```

Enter the password when prompted.

Inside `psql`:

``` sql
SELECT current_database(), current_user;
```

List tables:

``` sql
\dt
```

Check migration version:

``` sql
SELECT * FROM alembic_version;
```

The recorded main tracking database revision was:

``` text
bda7b8c39065
```

Check the database user's `CREATEDB` attribute, if connected with
sufficient privileges:

``` sql
SELECT rolname, rolcreatedb
FROM pg_roles
WHERE rolname = 'mlflowadmin';
```

The recorded result showed `rolcreatedb = true`.

### 6.3 Tables observed

The metadata schema contained tables including:

-   `experiments`
-   `experiment_tags`
-   `runs`
-   `metrics`
-   `params`
-   `tags`
-   `datasets`
-   `inputs`
-   `input_tags`
-   `latest_metrics`
-   `logged_models`
-   `logged_model_metrics`
-   `logged_model_params`
-   `logged_model_tags`
-   `model_versions`
-   `registered_models`
-   `registered_model_aliases`
-   `registered_model_tags`
-   `model_version_tags`
-   `trace_info`
-   `trace_request_metadata`
-   `trace_tags`
-   `alembic_version`

This confirmed that the main MLflow tracking schema had been created.
The list is not necessarily exhaustive.

Exit `psql`:

``` sql
\q
```

------------------------------------------------------------------------

## 7. Separate authentication database

The two database names have different purposes:

  Database          Purpose
  ----------------- ----------------------------------
  `mlflowdb`        MLflow tracking metadata
  `mlflow_authdb`   Basic Auth users and permissions

We created `mlflow_authdb` successfully and changed the authentication
database URI in `/etc/mlflow/basic_auth.ini` to:

``` text
postgresql://mlflowadmin:<DB_PASSWORD>@mlflow-metadata-db.ckh0wmiwcs1t.us-east-1.rds.amazonaws.com:5432/mlflow_authdb?sslmode=require
```

The recorded authentication migration revision was:

``` text
8606fa83a998
```

Connect to the authentication database:

``` bash
psql "host=mlflow-metadata-db.ckh0wmiwcs1t.us-east-1.rds.amazonaws.com port=5432 dbname=mlflow_authdb user=mlflowadmin sslmode=require"
```

Then:

``` sql
SELECT current_database(), current_user;
SELECT * FROM alembic_version;
\q
```

The exact original SQL command used to create `mlflow_authdb` is not
preserved in the available record. If the database already exists, do
not attempt to recreate it. If creating it on a new setup, run
`CREATE DATABASE mlflow_authdb;` while connected to a different database
as a PostgreSQL role permitted to create databases.

------------------------------------------------------------------------

## 8. MLflow Basic Auth configuration

The file was:

``` text
/etc/mlflow/basic_auth.ini
```

The final recorded relevant settings were:

``` ini
[mlflow]
admin_username = admin
default_permission = MANAGE
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
```

The same configuration file's database URI was changed to point to
`mlflow_authdb`, not `mlflowdb`.

Edit:

``` bash
sudo nano /etc/mlflow/basic_auth.ini
```

Check the non-secret settings:

``` bash
sudo grep -E '^(admin_username|default_permission|authorization_function)' \
  /etc/mlflow/basic_auth.ini
```

Inspect the file with the database password redacted:

``` bash
sudo sed -E 's#(postgresql://[^:]+:)[^@]+@#\1[REDACTED]@#' \
  /etc/mlflow/basic_auth.ini
```

### 8.1 Administrator password error

The first authentication initialization attempt failed with:

``` text
Password must be a string longer than 12 characters.
```

The admin password did not satisfy the installed MLflow version's
requirement. We corrected it and initialized successfully, with output
equivalent to:

``` text
Created admin user 'admin'.
Authentication app created successfully
```

The exact initialization command is not fully preserved in the setup
record. Do not rerun an initialization routine blindly on an existing
authentication database; first inspect the installed MLflow version and
existing user state. MLflow's Basic Auth feature was also reported as
experimental in that installed release.

### 8.2 Permission levels

  Permission           Read   Update   Delete   Manage
  ------------------ ------ -------- -------- --------
  `READ`                Yes       No       No       No
  `EDIT`                Yes      Yes       No       No
  `MANAGE`              Yes      Yes      Yes      Yes
  `NO_PERMISSIONS`       No       No       No       No

The initial default was `READ`. Later we changed it to `MANAGE` after
investigating the experiment creation behavior.

**Security consideration:** `MANAGE` is broad. It does not turn a
regular user into an administrator, and user-management endpoints have
separate checks. Reassess this setting before allowing multiple users or
sensitive data.

------------------------------------------------------------------------

## 9. MLflow environment file

The environment file was:

``` text
/etc/mlflow/mlflow.env
```

The main service command used these environment variables:

``` ini
MLFLOW_BACKEND_STORE_URI=postgresql://mlflowadmin:<DB_PASSWORD>@mlflow-metadata-db.ckh0wmiwcs1t.us-east-1.rds.amazonaws.com:5432/mlflowdb?sslmode=require
MLFLOW_ARTIFACT_ROOT=s3://amrith-mlflow-artifacts/mlflow-artifacts
MLFLOW_FLASK_SERVER_SECRET_KEY=<LONG_RANDOM_SECRET>
```

These names are chosen to match the complete service command in this
runbook. **Check the live files before editing:** an earlier summary
used `MLFLOW_DEFAULT_ARTIFACT_ROOT`, while the later reconstructed
service used `MLFLOW_ARTIFACT_ROOT`. Those names are not
interchangeable. The service and environment file must agree.

Edit:

``` bash
sudo nano /etc/mlflow/mlflow.env
```

Generate a random secret if needed:

``` bash
openssl rand -hex 32
```

Restrict permissions because this file contains credentials:

``` bash
sudo chown root:root /etc/mlflow/mlflow.env
sudo chmod 600 /etc/mlflow/mlflow.env
```

Check variable names without exposing the password:

``` bash
sudo grep -E '^(MLFLOW_BACKEND_STORE_URI|MLFLOW_ARTIFACT_ROOT|MLFLOW_DEFAULT_ARTIFACT_ROOT|MLFLOW_FLASK_SERVER_SECRET_KEY)=' \
  /etc/mlflow/mlflow.env \
  | sed -E 's#(postgresql://[^:]+:)[^@]+@#\1[REDACTED]@#'
```

Do not use `cat` on the raw environment file when sharing terminal
output.

------------------------------------------------------------------------

## 10. systemd service configuration

Main service file:

``` text
/etc/systemd/system/mlflow.service
```

The complete command reconstructed from the recorded settings is:

``` ini
[Unit]
Description=MLflow Tracking Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
EnvironmentFile=/etc/mlflow/mlflow.env
WorkingDirectory=/home/ec2-user
ExecStart=/home/ec2-user/.local/bin/mlflow server --app-name basic-auth --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} --host 0.0.0.0 --port 5000 --workers 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

The original pasted `ExecStart` line was truncated after `--backend->`.
The command above is reconstructed from the saved setup details; inspect
the currently installed file before replacing it.

Edit:

``` bash
sudo nano /etc/systemd/system/mlflow.service
```

Inspect the effective service and any drop-in overrides:

``` bash
sudo systemctl cat mlflow
```

### 10.1 systemd drop-in override

The override file was:

``` text
/etc/systemd/system/mlflow.service.d/override.conf
```

The override included an `ExecStartPre` check to ensure
`MLFLOW_FLASK_SERVER_SECRET_KEY` was present before starting MLflow. The
exact full override text is not preserved in the setup record.

Inspect it:

``` bash
sudo ls -la /etc/systemd/system/mlflow.service.d/
sudo cat /etc/systemd/system/mlflow.service.d/override.conf
```

Do not replace the override with an invented version; use the current
file as the source of truth.

### 10.2 Reload and restart

After changing the unit or override:

``` bash
sudo systemctl daemon-reload
sudo systemctl restart mlflow
sudo systemctl status mlflow --no-pager
```

Enable the service to start at boot:

``` bash
sudo systemctl enable mlflow
```

Check whether it is enabled:

``` bash
sudo systemctl is-enabled mlflow
```

The recorded final service status was:

``` text
Active: active (running)
```

------------------------------------------------------------------------

## 11. Troubleshoot service startup

During setup, Gunicorn workers failed to boot. We stopped/restarted the
service, reset the failed state, inspected logs, and tested
authentication initialization separately. The password-length error was
corrected.

Useful commands:

``` bash
sudo systemctl status mlflow --no-pager
sudo journalctl -u mlflow -n 100 --no-pager
sudo journalctl -u mlflow -f
sudo systemctl show mlflow --property=EnvironmentFiles
sudo ss -ltnp | grep ':5000'
```

If the service is stuck in a failed state, after inspecting the logs:

``` bash
sudo systemctl stop mlflow
sudo systemctl reset-failed mlflow
sudo systemctl start mlflow
sudo systemctl status mlflow --no-pager
```

Check that the backend URI, artifact-root variable, and Flask secret are
present and correctly named. Avoid printing secrets to the terminal when
capturing logs.

------------------------------------------------------------------------

## 12. Test the tracking server with curl

Run these tests on EC2.

### 12.1 Unauthenticated root

``` bash
curl -s -o /dev/null -w "HTTP status: %{http_code}\n" \
  http://127.0.0.1:5000/
```

Recorded result:

``` text
HTTP status: 401
```

This is expected with Basic Auth enabled.

### 12.2 Authenticated root

Prompt for the password instead of embedding it in the command:

``` bash
read -r -p "MLflow username: " MLFLOW_USER
read -r -s -p "MLflow password: " MLFLOW_PASSWORD
echo

curl -sS -o /dev/null -w "HTTP status: %{http_code}\n" \
  -u "$MLFLOW_USER:$MLFLOW_PASSWORD" \
  http://127.0.0.1:5000/
```

Recorded authenticated result:

``` text
HTTP status: 200
```

### 12.3 Correct experiments endpoint

The endpoint `/api/2.0/mlflow/experiments/list` returned `404`. The
successful endpoint was:

``` text
POST /api/2.0/mlflow/experiments/search
```

Run:

``` bash
curl -sS -u "$MLFLOW_USER:$MLFLOW_PASSWORD" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{"max_results":100}' \
  http://127.0.0.1:5000/api/2.0/mlflow/experiments/search
```

Recorded result: HTTP `200`, with the Default experiment, ID `0`, and
artifact URI:

``` text
s3://amrith-mlflow-artifacts/mlflow-artifacts/0
```

An empty JSON object had returned `400` because `max_results` was
missing or zero.

### 12.4 Create an experiment (creates a real resource)

``` bash
curl -sS -u "$MLFLOW_USER:$MLFLOW_PASSWORD" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{"name":"api-test-experiment"}' \
  http://127.0.0.1:5000/api/2.0/mlflow/experiments/create
```

Use a unique experiment name when rerunning this test.

Clear shell variables when done:

``` bash
unset MLFLOW_USER MLFLOW_PASSWORD
```

------------------------------------------------------------------------

## 13. Create and test the regular user

A non-admin user named `viewer_test` was created through:

``` text
POST /api/2.0/mlflow/users/create
```

The request was made using Python `requests` and `getpass` so passwords
were entered interactively. The result was HTTP `200`, user ID `2`,
`is_admin: false`.

A safe pattern for a Python client is:

``` python
import getpass
import requests

base_url = "http://127.0.0.1:5000"
admin_username = input("Admin username: ")
admin_password = getpass.getpass("Admin password: ")
new_username = input("New username: ")
new_password = getpass.getpass("New user's password: ")

response = requests.post(
    f"{base_url}/api/2.0/mlflow/users/create",
    auth=(admin_username, admin_password),
    json={"username": new_username, "password": new_password},
    timeout=30,
)

print("HTTP status:", response.status_code)
print(response.text)
response.raise_for_status()
```

This example uses the recorded API endpoint and interactive password
entry. Only run it if the user does not already exist; otherwise, handle
the API's duplicate-user response.

### 13.1 Permission investigation

We tested with `viewer_test`:

  -----------------------------------------------------------------------
  Operation                           Recorded result
  ----------------------------------- -----------------------------------
  Search experiments                  HTTP `200`

  Create experiment with initial      HTTP `200`, experiment ID `1`
  `READ` default                      

  Create experiment after default     HTTP `200`, experiment ID `2`
  changed to `MANAGE`                 
  -----------------------------------------------------------------------

We inspected installed source files under:

``` text
/home/ec2-user/.local/lib/python3.9/site-packages/mlflow/server/auth/
```

Files inspected included:

``` text
__init__.py
permissions.py
routes.py
```

The code showed that the user who creates an experiment is assigned
`MANAGE` permission for that experiment. This helps explain why global
defaults and per-experiment permission records do not always tell the
entire story.

The permission lookup endpoint tested was:

``` text
GET /api/2.0/mlflow/experiments/permissions/get
```

A request without `username` returned `400`; with `username=viewer_test`
for experiment `0`, it returned `404 RESOURCE_DOES_NOT_EXIST` because
the explicit permission record was not found.

The final configuration was changed to:

``` ini
default_permission = MANAGE
```

The service was restarted and the regular-user experiment-creation test
returned `200`. This did not prove every model registry, update, delete,
and permission-management operation was working.

------------------------------------------------------------------------

## 14. SSH tunnel for local access

A recorded WSL command used a local port `5001` forwarded to EC2 port
`5000`:

``` bash
ssh -i ~/.ssh/mlflow-server-key.pem \
  -N -L 5001:127.0.0.1:5000 \
  ec2-user@13.221.201.200
```

The public IP `13.221.201.200` was present in the recorded command.
Verify it is still the instance's current address before relying on it.
The key path must also exist on your machine.

Keep this terminal open. In another terminal:

``` bash
curl -i http://127.0.0.1:5001/
```

An unauthenticated `401` means the request reached MLflow and Basic Auth
is active. It is not a failed tunnel.

For this tunnel, use the local tracking URI:

``` text
http://127.0.0.1:5001
```

You need SSH access through security group port `22`. You do not need to
open port `5000` to the public internet when using the tunnel.

------------------------------------------------------------------------

## 15. Local Python client setup and end-to-end test

The remaining milestone is to create a run from your local computer and
verify its metadata and artifact.

### 15.1 Install the matching MLflow client

``` bash
python -m pip install "mlflow==3.1.4"
python -c "import mlflow; print(mlflow.__version__)"
```

Use a Python environment compatible with this package version.

### 15.2 Set the tracking URI

For WSL/Bash, with the SSH tunnel active:

``` bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
export MLFLOW_TRACKING_USERNAME="admin"
read -r -s -p "MLflow password: " MLFLOW_TRACKING_PASSWORD
echo
export MLFLOW_TRACKING_PASSWORD
```

For a local client, use a valid MLflow username and password. Do not
save credentials in your project source.

### 15.3 Test script

Save the following as `test_mlflow.py`:

``` python
import os
import tempfile
from pathlib import Path

import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("local-client-connectivity-test")

with tempfile.TemporaryDirectory() as temp_dir:
    artifact = Path(temp_dir) / "test.txt"
    artifact.write_text(
        "Artifact uploaded through MLflow.",
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="local-client-test") as run:
        mlflow.log_param("client", "local")
        mlflow.log_param("target_mlflow_version", "3.1.4")
        mlflow.log_metric("test_accuracy", 0.95)
        mlflow.log_artifact(str(artifact))

        print("Run ID:", run.info.run_id)
        print("Experiment ID:", run.info.experiment_id)
        print("Artifact URI:", mlflow.get_artifact_uri())
```

Run:

``` bash
python test_mlflow.py
```

Verify: 1. The run appears in the MLflow UI. 2. The parameter and metric
appear in the run. 3. The text artifact appears in the run's artifacts.
4. The object is present under the expected S3 artifact location.

Then clear the password variable:

``` bash
unset MLFLOW_TRACKING_PASSWORD
```

If the test fails, inspect the EC2 service logs and confirm network
connectivity, credentials, RDS access, and S3 role permissions.

------------------------------------------------------------------------

## 16. Final verification checklist

### AWS

-   [ ] EC2 instance is running in `us-east-1`.
-   [ ] `MLflowArtifactBucketAccess` is attached to EC2.
-   [ ] S3 bucket is private and the artifact prefix is accessible by
    the role.
-   [ ] RDS PostgreSQL is reachable from EC2 on port `5432`.
-   [ ] RDS inbound rules allow the EC2 security group, not the entire
    internet.
-   [ ] EC2 inbound rules restrict SSH and, if used, direct MLflow
    access.

### EC2 and service

-   [ ] MLflow version is `3.1.4`.
-   [ ] `/etc/mlflow/mlflow.env` has the backend URI, artifact-root
    variable and Flask secret.
-   [ ] `/etc/mlflow/basic_auth.ini` references `mlflow_authdb`.
-   [ ] `/etc/systemd/system/mlflow.service` uses the correct
    environment-variable names.
-   [ ] The override file and `ExecStartPre` secret check are present as
    intended.
-   [ ] `sudo systemctl status mlflow` reports active/running.
-   [ ] MLflow is listening on port `5000`.

### Authentication and logging

-   [ ] Unauthenticated request returns `401`.
-   [ ] Authenticated request returns `200`.
-   [ ] Experiments search succeeds.
-   [ ] A regular user can perform only the operations intended by your
    permission policy.
-   [ ] A local client creates a run.
-   [ ] The run's metrics/parameters are visible.
-   [ ] The artifact upload is verified in S3.

------------------------------------------------------------------------

## 17. Important final state and items not yet proven

### Confirmed by the recorded setup

-   EC2 was running Amazon Linux 2023 and MLflow `3.1.4`.
-   The EC2 role could list and access the configured S3 prefix; a test
    file was visible.
-   The main PostgreSQL database contained MLflow tracking tables.
-   A separate `mlflow_authdb` database was created.
-   Basic Auth initialization succeeded after fixing the password-length
    error.
-   The systemd service reached `active (running)`.
-   An unauthenticated root request returned `401`.
-   An authenticated root request returned `200`.
-   Authenticated experiment search returned `200`.
-   A non-admin user was created.
-   The default permission was changed from `READ` to `MANAGE`.
-   Experiment creation succeeded after the final permission change.

### Still requiring verification

-   Complete local-to-server MLflow run logging.
-   Successful upload of an artifact through the MLflow client, rather
    than only a direct AWS CLI upload.
-   Model registry, deletion, and other permissions not covered by the
    recorded tests.
-   The current EC2 public IP/DNS, current security-group rules, and
    full S3/RDS console settings.
-   The exact current contents of `override.conf` and the original,
    untruncated `ExecStart` line.

**Bottom line:** the recorded setup has the tracking server, PostgreSQL
metadata storage, S3 artifact storage, systemd service, and Basic Auth
configured and API-tested. The next important proof is an end-to-end run
from your local machine with an artifact verified in S3.
