# MaaS 平台搭建 Command 手册

> **版本**: v0.1 | **状态**: 草稿  
> **对标架构**: maas-platform-full.md（百卡集群四层架构）  
> **目标**: 将架构文档中的设计落地为可执行的 Command 命令集  
> **范围**: K8s 集群部署、网络配置、存储搭建、GPU 纳管、训练/推理部署

---

## 目录

1. [环境准备 & 操作系统基线](#1-环境准备--操作系统基线)
2. [Kubernetes 集群部署](#2-kubernetes-集群部署)
3. [网络配置（四网分离 + RDMA + NCCL）](#3-网络配置四网分离--rdma--nccl)
4. [存储搭建（Ceph + 三级缓存）](#4-存储搭建ceph--三级缓存)
5. [GPU 纳管（NVIDIA GPU Operator + Device Plugin）](#5-gpu-纳管nvidia-gpu-operator--device-plugin)
6. [调度器部署（Volcano + Gang + Binpack）](#6-调度器部署volcano--gang--binpack)
7. [训练体系部署（DeepSpeed + 多机多卡）](#7-训练体系部署deepspeed--多机多卡)
8. [推理体系部署（vLLM + 网关）](#8-推理体系部署vllm--网关)
9. [可观测性（Prometheus + DCGM + Grafana + Loki）](#9-可观测性prometheus--dcgm--grafana--loki)
10. [安全加固（RBAC + 镜像扫描 + SSH 密钥）](#10-安全加固rbac--镜像扫描--ssh-密钥)
11. [日常运维 Command 速查](#11-日常运维-command-速查)

---

## 1. 环境准备 & 操作系统基线

### 1.1 操作系统安装

```bash
# 所有节点（训练/推理/CPU/存储）统一基线
# OS: Ubuntu 22.04 LTS / Rocky Linux 9.x
# 内核: ≥ 5.15
# NVIDIA 驱动: ≥ 535.104

# 检查内核版本
uname -r

# 设置 hostname（按节点角色编号）
# 训练节点: train-node-01 ~ train-node-50
# 推理节点: infer-node-01 ~ infer-node-25
# CPU 节点:  cpu-node-01  ~ cpu-node-15
# 存储节点:  stor-node-01 ~ stor-node-10
hostnamectl set-hostname train-node-01

# 关闭 swap（K8s 要求）
swapoff -a
sed -i '/swap/d' /etc/fstab

# 加载必要内核模块
cat >> /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF
modprobe overlay
modprobe br_netfilter

# 网络参数调优
cat >> /etc/sysctl.d/99-k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
fs.inotify.max_user_watches         = 524288
fs.inotify.max_user_instances       = 512
EOF
sysctl --system
```

### 1.2 安装 containerd

```bash
# 安装 containerd（K8s 推荐运行时）
apt-get update && apt-get install -y containerd

# 配置 containerd 使用 systemd cgroup driver
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml

# 修改 config.toml 中 SystemdCgroup = true
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml

systemctl restart containerd
systemctl enable containerd
```

### 1.3 安装 kubeadm/kubelet/kubectl

```bash
# 添加 Kubernetes apt 仓库
apt-get update
apt-get install -y apt-transport-https ca-certificates curl
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | \
  gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] \
  https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' | \
  tee /etc/apt/sources.list.d/kubernetes.list

apt-get update
apt-get install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl

systemctl enable kubelet
```

---

## 2. Kubernetes 集群部署

### 2.1 初始化控制面（Master 节点）

```bash
# 在第一个 Master 节点上初始化控制面
# --pod-network-cidr: Calico 默认网段
# --service-cidr: Service 网络
# --control-plane-endpoint: 高可用 VIP 或负载均衡器地址
kubeadm init \
  --control-plane-endpoint=k8s-api.maas.local:6443 \
  --pod-network-cidr=192.168.0.0/16 \
  --service-cidr=10.96.0.0/12 \
  --kubernetes-version=v1.28.4 \
  --upload-certs

# 初始化成功后，记录输出的 join 命令
# 配置 kubectl
mkdir -p $HOME/.kube
cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config

# 验证控制面组件
kubectl get pods -n kube-system
kubectl get nodes
```

### 2.2 加入 Worker 节点

```bash
# 在每个 Worker 节点（GPU/CPU/存储节点）上执行 join 命令
# 命令来自 kubeadm init 输出
kubeadm join k8s-api.maas.local:6443 \
  --token <token> \
  --discovery-token-ca-cert-hash sha256:<hash>

# GPU 节点添加 Taint（防止非 GPU 任务调度到 GPU 节点）
# 在 Master 上执行
kubectl taint nodes train-node-{01..50} gpu=true:NoSchedule

# 给节点打标签（按角色）
kubectl label nodes train-node-{01..50} node-role=training
kubectl label nodes infer-node-{01..25} node-role=inference
kubectl label nodes cpu-node-{01..15}   node-role=data-processing
kubectl label nodes stor-node-{01..10}  node-role=storage
```

### 2.3 部署 Calico CNI（BGP 模式，无 Overlay）

```bash
# 下载并部署 Calico（BGP 模式，推荐 GPU 集群使用）
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.4/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.4/manifests/custom-resources.yaml

# 验证 Calico 状态
kubectl get pods -n calico-system
calicoctl node status   # 需要安装 calicoctl

# 确认 BGP Peering 状态
kubectl exec -n calico-system -it $(kubectl get pods -n calico-system -l k8s-app=calico-node -o jsonpath='{.items[0].metadata.name}') -- calicoctl node status
```

### 2.4 安装 Helm

```bash
# Helm 是后续所有组件部署的基础工具
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

---

## 3. 网络配置（四网分离 + RDMA + NCCL）

### 3.1 四网物理分离配置

```bash
# 每台训练节点配置 5 张网卡的角色分配：
# eth0  - 管理网 (10G)     - K8s/SSH/监控
# ib0   - 计算网主 (200G IB) - NCCL AllReduce
# ib1   - 计算网备 (200G IB) - NCCL 冗余
# eth2  - 存储网主 (100G)    - 数据读写
# eth3  - 存储网备 (100G)    - 存储冗余

# ============================================
# 3.1.1 管理网配置（eth0）
# ============================================
cat > /etc/netplan/01-management.yaml <<EOF
network:
  version: 2
  ethernets:
    eth0:
      addresses: [10.0.1.10/24]
      gateway4: 10.0.1.1
      nameservers:
        addresses: [10.0.1.2]
      mtu: 1500
EOF
netplan apply

# ============================================
# 3.1.2 存储网 Bond 配置（eth2+eth3 → bond1）
# ============================================
cat > /etc/netplan/02-storage.yaml <<EOF
network:
  version: 2
  bonds:
    bond1:
      interfaces: [eth2, eth3]
      parameters:
        mode: 802.3ad          # LACP
        mii-monitor-interval: 100
        transmit-hash-policy: layer3+4
      addresses: [10.0.2.10/24]
      routes:
        - to: 10.0.2.0/24
          via: 10.0.2.1
      mtu: 9000                # Jumbo Frame
  ethernets:
    eth2: {}
    eth3: {}
EOF
netplan apply

# 验证 bond 状态
cat /proc/net/bonding/bond1

# ============================================
# 3.1.3 策略路由：不同流量走不同网卡
# ============================================
# 存储流量走存储网
ip rule add from 10.0.2.0/24 table storage
ip route add default via 10.0.2.1 dev bond1 table storage

# 管理流量走管理网（已有默认路由）
ip rule add from 10.0.1.0/24 table management
ip route add default via 10.0.1.1 dev eth0 table management

# 持久化策略路由
cat > /etc/iproute2/rt_tables <<EOF
100 management
200 storage
EOF
```

### 3.2 InfiniBand / RoCE 驱动安装

```bash
# ============================================
# 3.2.1 安装 Mellanox OFED 驱动（RoCE 场景）
# ============================================
# 下载 MOFED（推荐 ≥ 5.8）
wget https://content.mellanox.com/ofed/MLNX_OFED-5.8-1.1.2.1/MLNX_OFED_LINUX-5.8-1.1.2.1-ubuntu22.04-x86_64.tgz
tar -xzf MLNX_OFED_LINUX-5.8-1.1.2.1-ubuntu22.04-x86_64.tgz
cd MLNX_OFED_LINUX-5.8-1.1.2.1-ubuntu22.04-x86_64

# 安装（包含 RDMA 核心驱动）
./mlnxofedinstall --all --force

# 加载 RDMA 内核模块
cat >> /etc/modules-load.d/rdma.conf <<EOF
rdma_cm
ib_umad
ib_uverbs
ib_ipoib
mlx5_ib
nvidia_peermem    # GPUDirect RDMA 关键模块
EOF

# 重启加载
systemctl restart openibd
reboot

# ============================================
# 3.2.2 验证 RDMA 设备
# ============================================
# 检查 IB/RoCE 设备状态
ibv_devinfo
# 期望输出:
#   hca_id: mlx5_0
#     transport: InfiniBand (0)
#     fw_ver: 28.40.1002
#     node_guid: ...
#     sys_image_guid: ...
#     vendor_id: 0x02c9 (Mellanox)
#     vendor_part_id: 4123
#     hw_ver: 0x0
#     board_id: ...
#     phys_port_cnt: 1
#       port: 1
#         state: PORT_ACTIVE (4)
#         max_mtu: 4096 (5)
#         active_mtu: 4092 (4)
#         sm_lid: ...
#         port_lid: ...
#         port_lmc: 0x00
#         link_layer: InfiniBand

# 查看 IB 设备状态
ibstat

# 网卡与 IB 设备映射
ibdev2netdev
# 期望输出:
#   mlx5_0 port 1 ==> ib0 (Up)
#   mlx5_1 port 1 ==> ib1 (Up)
```

### 3.3 GPUDirect RDMA 配置

```bash
# ============================================
# 3.3.1 验证 GPUDirect RDMA 是否可用
# ============================================
# 检查 nvidia_peermem 内核模块
lsmod | grep nvidia_peermem
# 期望输出: nvidia_peermem  xxxxxx  0

# 如果没有加载，手动加载
modprobe nvidia_peermem

# 检查 NCCL 是否能检测到 RDMA
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET,INIT python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
" 2>&1 | grep -iE 'rdma|ib|net'

# 期望输出: "NET/IB : Using [0] mlx5_0:1/IB"

# ============================================
# 3.3.2 NCCL 环境变量配置
# ============================================
cat > /etc/profile.d/nccl-env.sh <<'EOF'
# NCCL 使用计算网卡（IB 接口）
export NCCL_SOCKET_IFNAME=ib0,ib1
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3          # GPU Direct RDMA
export NCCL_ALGO=Tree                # Tree 算法（高带宽场景）
export NCCL_CROSS_NIC=0              # 禁止跨网卡
export NCCL_IB_GID_INDEX=3           # RoCE v2 GID
export NCCL_P2P_LEVEL=NVL            # P2P 走 NVLink
export NCCL_IB_QPS_PER_CONNECTION=4  # 每连接 QPS 数
export NCCL_IB_TC=136                # IB Traffic Class
export NCCL_IB_TIMEOUT=22            # IB Timeout
export NCCL_DEBUG=WARN               # 生产环境使用 WARN，调试时用 INFO
EOF

source /etc/profile.d/nccl-env.sh
```

### 3.4 MTU 一致性校验

```bash
# ============================================
# 3.4.1 检查各接口 MTU
# ============================================
echo "=== MTU 一致性检查 ==="
echo -e "接口\t\t角色\t\t当前MTU\t预期MTU\t状态"
echo "-----------------------------------------------"

check_mtu() {
  local iface=$1
  local role=$2
  local expected=$3
  local actual=$(cat /sys/class/net/$iface/mtu 2>/dev/null || echo "N/A")
  local status="OK"
  if [ "$actual" != "$expected" ]; then
    status="MISMATCH"
  fi
  echo -e "${iface}\t\t${role}\t\t${actual}\t${expected}\t${status}"
}

check_mtu "eth0"  "管理网"  "1500"
check_mtu "ib0"   "计算网(主)" "4092"
check_mtu "ib1"   "计算网(备)" "4092"
check_mtu "bond1" "存储网"  "9000"

# ============================================
# 3.4.2 端到端 MTU 测试
# ============================================
# IB 网络 MTU 测试（4092 - 28 ICMP header = 4064）
ping -M do -s 4064 <对端IB_IP>

# 存储网 Jumbo Frame 测试（9000 - 28 = 8972）
ping -M do -s 8972 <对端存储IP>

# ============================================
# 3.4.3 修复 MTU 不匹配
# ============================================
# 临时修复
ip link set ib0 mtu 4092
ip link set ib1 mtu 4092
ip link set bond1 mtu 9000

# 永久修复：在 netplan 配置中添加 mtu 参数（见 3.1 节）
```

### 3.5 NCCL 性能基准测试

```bash
# ============================================
# 3.5.1 安装 nccl-tests
# ============================================
apt-get install -y build-essential libibverbs-dev libnccl-dev
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j MPI=1

# ============================================
# 3.5.2 单机 8 卡 AllReduce 测试（NVLink 域内）
# ============================================
./build/all_reduce_perf -b 8 -e 4G -f 2 -g 8 -c 1

# 期望结果（NVLink）:
#   Size: 4G    Time: ~5ms    Alg: Tree   BW: ~850 GB/s

# ============================================
# 3.5.3 多机 AllReduce 测试（IB 跨节点）
# ============================================
# 在 2 台机器上分别运行
# 机器 A (Rank 0):
mpirun -np 16 -hostfile hostfile \
  -x NCCL_DEBUG=INFO \
  ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8 -c 1

# 期望结果（200G IB）:
#   Size: 1G    Time: ~50ms   Alg: Tree   BW: ~23 GB/s

# ============================================
# 3.5.4 RDMA 带宽测试（ib_write_bw）
# ============================================
# Server 端
ib_write_bw -d mlx5_0

# Client 端
ib_write_bw -d mlx5_0 <server_ip>

# 期望结果（200G IB）: ~23 GB/s (理论线速的 ~90%)
```

### 3.6 Calico NetworkPolicy 配置

```bash
# ============================================
# 3.6.1 训练 Pod 网络隔离
# ============================================
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: training-pod-isolation
  namespace: maas-training
spec:
  podSelector:
    matchLabels:
      app: training-job
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # 允许同训练任务的 Pod 间 NCCL 通信
    - from:
        - podSelector:
            matchLabels:
              app: training-job
      ports:
        - protocol: TCP
          port: 23456   # NCCL 默认端口
  egress:
    # 允许访问存储网络
    - to:
        - ipBlock:
            cidr: 10.0.2.0/24   # 存储网段
    # 允许访问 K8s API
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 443
    # 允许 DNS
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
EOF

# ============================================
# 3.6.2 推理服务网络策略
# ============================================
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: inference-isolation
  namespace: maas-inference
spec:
  podSelector:
    matchLabels:
      app: inference-service
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # 只允许网关访问推理服务
    - from:
        - podSelector:
            matchLabels:
              app: inference-gateway
      ports:
        - protocol: TCP
          port: 8000
  egress:
    # 允许访问模型存储
    - to:
        - ipBlock:
            cidr: 10.0.2.0/24
    # 允许 DNS
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
      ports:
        - protocol: UDP
          port: 53
EOF
```

---

## 4. 存储搭建（Ceph + 三级缓存）

### 4.1 Rook-Ceph 部署

```bash
# ============================================
# 4.1.1 添加 Rook Helm 仓库
# ============================================
helm repo add rook-release https://charts.rook.io/release
helm repo update

# ============================================
# 4.1.2 部署 Rook Operator
# ============================================
helm install --create-namespace --namespace rook-ceph rook-ceph rook-release/rook-ceph \
  --version v1.13.5

# ============================================
# 4.1.3 创建 Ceph 集群
# ============================================
kubectl apply -f - <<EOF
apiVersion: ceph.rook.io/v1
kind: CephCluster
metadata:
  name: maas-storage
  namespace: rook-ceph
spec:
  cephVersion:
    image: quay.io/ceph/ceph:v18.2.1    # Reef 版本
  dataDirHostPath: /var/lib/rook
  mon:
    count: 3
    allowMultiplePerNode: false
  mgr:
    count: 2
  storage:
    useAllNodes: false
    useAllDevices: false
    # 指定存储节点的裸设备
    nodes:
      - name: stor-node-01
        devices:
          - name: /dev/nvme0n1
          - name: /dev/nvme1n1
      - name: stor-node-02
        devices:
          - name: /dev/nvme0n1
          - name: /dev/nvme1n1
      - name: stor-node-03
        devices:
          - name: /dev/nvme0n1
          - name: /dev/nvme1n1
  # 三副本策略
  disruptionManagement:
    managePodBudgets: true
    osdMaintenanceTimeout: 30
EOF

# 等待 Ceph 集群就绪
kubectl -n rook-ceph get pods --watch
# 期望看到: mon.*, mgr.*, osd.* 全部 Running

# ============================================
# 4.1.4 验证 Ceph 集群状态
# ============================================
# 使用 ceph-toolbox 进入 Ceph 集群
kubectl apply -f https://raw.githubusercontent.com/rook/rook/master/deploy/examples/toolbox.yaml
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- ceph status

# 期望输出:
#   cluster:
#     id:     xxx
#     health: HEALTH_OK
#   services:
#     mon: 3 daemons, quorum a,b,c
#     mgr: a(active), standbys: b
#     osd: 6 osds: 6 up, 6 in
#   data:
#     pools:   3 pools, 96 pgs
#     objects: 0 objects, 0 B
#     usage:   6.0 GiB used, 3.5 TiB / 3.5 TiB avail
#     pgs:     96 active+clean
```

### 4.2 创建 CephFS 存储池（L2 并行文件系统）

```bash
# ============================================
# 4.2.1 创建 CephFS 文件系统
# ============================================
kubectl apply -f - <<EOF
apiVersion: ceph.rook.io/v1
kind: CephFilesystem
metadata:
  name: maas-fs
  namespace: rook-ceph
spec:
  metadataPool:
    replicated:
      size: 3
  dataPools:
    - failureDomain: host
      replicated:
        size: 3
      name: data-pool
  metadataServer:
    activeCount: 1
    activeStandby: true
    resources:
      limits:
        cpu: "4"
        memory: "8Gi"
      requests:
        cpu: "2"
        memory: "4Gi"
EOF

# ============================================
# 4.2.2 创建 StorageClass
# ============================================
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: cephfs-maas
provisioner: rook-ceph.cephfs.csi.ceph.com
parameters:
  clusterID: rook-ceph
  fsName: maas-fs
  pool: maas-fs-data-pool
  csi.storage.k8s.io/provisioner-secret-name: rook-csi-cephfs-provisioner
  csi.storage.k8s.io/provisioner-secret-namespace: rook-ceph
  csi.storage.k8s.io/controller-expand-secret-name: rook-csi-cephfs-provisioner
  csi.storage.k8s.io/controller-expand-secret-namespace: rook-ceph
  csi.storage.k8s.io/node-stage-secret-name: rook-csi-cephfs-node
  csi.storage.k8s.io/node-stage-secret-namespace: rook-ceph
reclaimPolicy: Retain
allowVolumeExpansion: true
EOF

# ============================================
# 4.2.3 创建 CephFS PVC 并挂载到训练节点
# ============================================
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: maas-training-data
  namespace: maas-training
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: cephfs-maas
  resources:
    requests:
      storage: 500Gi
EOF

# 在训练 Pod 中挂载
# volumeMounts:
#   - name: training-data
#     mountPath: /mnt/cephfs/datasets
# volumes:
#   - name: training-data
#     persistentVolumeClaim:
#       claimName: maas-training-data
```

### 4.3 Ceph RBD 存储池（L1 本地 NVMe 缓存）

```bash
# ============================================
# 4.3.1 创建 RBD 存储池
# ============================================
kubectl apply -f - <<EOF
apiVersion: ceph.rook.io/v1
kind: CephBlockPool
metadata:
  name: maas-rbd
  namespace: rook-ceph
spec:
  failureDomain: host
  replicated:
    size: 3
  parameters:
    compression_mode: aggressive
EOF

# ============================================
# 4.3.2 创建 RBD StorageClass（高性能 SSD）
# ============================================
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ceph-rbd-nvme
provisioner: rook-ceph.rbd.csi.ceph.com
parameters:
  clusterID: rook-ceph
  pool: maas-rbd
  imageFormat: "2"
  imageFeatures: layering
  csi.storage.k8s.io/provisioner-secret-name: rook-csi-rbd-provisioner
  csi.storage.k8s.io/provisioner-secret-namespace: rook-ceph
  csi.storage.k8s.io/controller-expand-secret-name: rook-csi-rbd-provisioner
  csi.storage.k8s.io/controller-expand-secret-namespace: rook-ceph
  csi.storage.k8s.io/node-stage-secret-name: rook-csi-rbd-node
  csi.storage.k8s.io/node-stage-secret-namespace: rook-ceph
  csi.storage.k8s.io/fstype: ext4
reclaimPolicy: Delete
allowVolumeExpansion: true
EOF
```

### 4.4 MinIO 对象存储（L3 冷热归档）

```bash
# ============================================
# 4.4.1 部署 MinIO（Helm）
# ============================================
helm repo add minio https://operator.min.io
helm repo update

kubectl create namespace minio-operator
helm install --namespace minio-operator minio-operator minio/operator

# 创建租户
kubectl apply -f - <<EOF
apiVersion: minio.min.io/v2
kind: Tenant
metadata:
  name: maas-minio
  namespace: minio
spec:
  pools:
    - servers: 4
      volumesPerServer: 4
      volumeClaimTemplate:
        spec:
          storageClassName: ceph-rbd-nvme
          accessModes:
            - ReadWriteOnce
          resources:
            requests:
              storage: 1Ti
  certificate:
    requestAutoCert: true
  credsSecret:
    name: maas-minio-secret
EOF

# ============================================
# 4.4.2 创建存储桶
# ============================================
# 使用 mc (MinIO Client)
mc alias set maas https://minio.maas.local <access-key> <secret-key>
mc mb maas/training-data
mc mb maas/model-weights
mc mb maas/checkpoints
mc mb maas/logs-archive

# 设置生命周期策略（冷热分层）
mc ilm add maas/training-data \
  --transition-days 30 \
  --transition-tier "COLD"
```

### 4.5 三级缓存预热命令

```bash
# ============================================
# 4.5.1 数据集从 L3 → L2 → L1 预热
# ============================================
# 从 MinIO (L3) 下载到 CephFS (L2)
mc cp --recursive maas/training-data/c4-en-v1.1/ /mnt/cephfs/datasets/c4-en-v1.1/

# 从 CephFS (L2) 预热到本地 NVMe (L1)
# 使用 vmtouch 将文件加载到 page cache
apt-get install -y vmtouch
vmtouch -t /data/nvme/cache/c4-en-v1.1/

# 验证缓存命中
vmtouch /data/nvme/cache/c4-en-v1.1/
# 输出: Files: 1234, Directories: 56
#       Resident Pages: 50000/50000  200M/200M  100%
#       Elapsed: 0.001 seconds

# ============================================
# 4.5.2 Ceph 扩容限流（防止数据迁移影响训练）
# ============================================
# 新增 OSD 前，限制回填速度
ceph config set osd osd_max_backfills 1
ceph config set osd osd_recovery_max_active 1
ceph config set osd osd_recovery_op_priority 3

# 分批扩容（每次 2-3 个 OSD，不要一次性全加）
# 扩容完成后恢复正常
ceph config set osd osd_max_backfills 4
ceph config set osd osd_recovery_max_active 3
ceph config set osd osd_recovery_op_priority 10
```

---

## 5. GPU 纳管（NVIDIA GPU Operator + Device Plugin）

### 5.1 安装 NVIDIA GPU Operator

```bash
# ============================================
# 5.1.1 添加 NVIDIA Helm 仓库
# ============================================
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
  && helm repo update

# ============================================
# 5.1.2 创建 GPU Operator Namespace
# ============================================
kubectl create namespace gpu-operator

# ============================================
# 5.1.3 部署 GPU Operator
# ============================================
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --version v23.9.1 \
  --set driver.version=535.104 \
  --set toolkit.version=1.14.3 \
  --set devicePlugin.version=0.14.1 \
  --set dcgmExporter.version=3.2.4 \
  --set driver.enabled=true \
  --set mig.strategy=none

# ============================================
# 5.1.4 验证 GPU Operator 组件
# ============================================
kubectl get pods -n gpu-operator --watch
# 期望看到以下组件全部 Running:
#   gpu-operator-xxxxx                          1/1 Running
#   node-feature-discovery-master-xxx           1/1 Running
#   node-feature-discovery-worker-xxx           1/1 Running  (每个节点)
#   nvidia-driver-daemonset-xxx                 1/1 Running  (每个 GPU 节点)
#   nvidia-container-toolkit-daemonset-xxx      1/1 Running  (每个 GPU 节点)
#   nvidia-device-plugin-daemonset-xxx          1/1 Running  (每个 GPU 节点)
#   dcgm-exporter-xxx                           1/1 Running  (每个 GPU 节点)

# ============================================
# 5.1.5 验证 GPU 资源注册
# ============================================
# 检查节点 GPU 资源
kubectl describe node train-node-01 | grep -A 5 "Allocatable:"
# 期望输出:
#   Allocatable:
#     nvidia.com/gpu:     8

# 运行 GPU 验证 Pod
kubectl run cuda-vector-add \
  --image=nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda11.8 \
  --restart=Never --limits=nvidia.com/gpu=1
kubectl logs cuda-vector-add
# 期望输出: Result = PASS
```

### 5.2 GPU Feature Discovery（拓扑标签注入）

```bash
# GPU Operator 默认包含 GPU Feature Discovery
# 验证拓扑标签是否已注入
kubectl get nodes train-node-01 --show-labels | tr ',' '\n' | grep nvidia
# 期望输出:
#   nvidia.com/gpu.product=NVIDIA-A800-SXM4-80GB
#   nvidia.com/gpu.count=8
#   nvidia.com/nvlink.topology=ring-8
#   nvidia.com/cuda.driver.version=535.104
#   nvidia.com/cuda.runtime.version=12.2

# 查看 GPU 拓扑（机内 NVLink 关系）
kubectl exec -n gpu-operator -it \
  $(kubectl get pods -n gpu-operator -l app=nvidia-device-plugin-daemonset -o jsonpath='{.items[0].metadata.name}') \
  -- nvidia-smi topo -m
```

### 5.3 NVIDIA Container Runtime 配置

```bash
# GPU Operator 自动配置 containerd 使用 nvidia 运行时
# 验证配置
cat /etc/containerd/config.toml | grep -A 5 nvidia
# 期望输出:
#   [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
#     runtime_type = "io.containerd.runc.v2"
#     [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
#       BinaryName = "/usr/bin/nvidia-container-runtime"

# 手动测试容器内 GPU 访问
kubectl run gpu-test --image=nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04 \
  --restart=Never --limits=nvidia.com/gpu=1 --command -- nvidia-smi
kubectl logs gpu-test
# 期望输出: nvidia-smi 正常显示 8 张 GPU 信息
```

---

## 6. 调度器部署（Volcano + Gang + Binpack）

### 6.1 安装 Volcano 调度器

```bash
# ============================================
# 6.1.1 添加 Volcano Helm 仓库
# ============================================
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update

# ============================================
# 6.1.2 部署 Volcano
# ============================================
kubectl create namespace volcano-system
helm install volcano volcano-sh/volcano \
  --namespace volcano-system \
  --version v1.8.0

# ============================================
# 6.1.3 验证 Volcano 组件
# ============================================
kubectl get pods -n volcano-system
# 期望看到:
#   volcano-admission-xxx    1/1 Running
#   volcano-controllers-xxx  1/1 Running
#   volcano-scheduler-xxx    1/1 Running
```

### 6.2 创建调度队列（Queue）

```bash
# ============================================
# 6.2.1 创建训练队列（高优先级）
# ============================================
kubectl apply -f - <<EOF
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: training-queue
spec:
  weight: 10
  capability:
    nvidia.com/gpu: 64
  reclaimable: true
  acl:
    allowUsers:
      - training-team
      - algorithm-team
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: inference-queue
spec:
  weight: 5
  capability:
    nvidia.com/gpu: 24
  reclaimable: false
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: experimental-queue
spec:
  weight: 2
  capability:
    nvidia.com/gpu: 16
  reclaimable: true       # 可被高优队列抢占
EOF
```

### 6.3 训练任务：Gang Scheduling + Binpack

```bash
# ============================================
# 6.3.1 训练任务定义（Gang Scheduling）
# ============================================
kubectl apply -f - <<EOF
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: training-job-001
  namespace: maas-training
spec:
  minMember: 8              # 最小 8 个 Pod 同时调度
  minResources:
    nvidia.com/gpu: 8
  queue: training-queue
  scheduleTimeoutSeconds: 300   # 5 分钟超时
---
apiVersion: batch/v1
kind: Job
metadata:
  name: deepspeed-training
  namespace: maas-training
  labels:
    app: training-job
spec:
  parallelism: 8
  completions: 8
  completionMode: Indexed
  template:
    metadata:
      labels:
        app: training-job
        queue: training-queue
      annotations:
        scheduling.volcano.sh/gang-name: training-job-001
    spec:
      schedulerName: volcano         # 使用 Volcano 调度器
      restartPolicy: Never
      tolerations:
        - key: gpu
          operator: Exists
          effect: NoSchedule
      nodeSelector:
        node-role: training
        nvidia.com/gpu.product: NVIDIA-A800-SXM4-80GB
      # 拓扑约束：尽量调度到同一节点
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: training-job
      containers:
        - name: trainer
          image: registry.maas.local/training:deepspeed-v1.0
          command: ["bash", "-c", "torchrun --nproc_per_node=1 train.py"]
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "64Gi"
              cpu: "16"
            requests:
              nvidia.com/gpu: 1
              memory: "64Gi"
              cpu: "16"
          volumeMounts:
            - name: training-data
              mountPath: /mnt/cephfs/datasets
            - name: checkpoint-dir
              mountPath: /mnt/checkpoints
      volumes:
        - name: training-data
          persistentVolumeClaim:
            claimName: maas-training-data
        - name: checkpoint-dir
          persistentVolumeClaim:
            claimName: maas-checkpoint-pvc
EOF
```

---

## 7. 训练体系部署（DeepSpeed + 多机多卡）

### 7.1 构建训练镜像

```bash
# ============================================
# 7.1.1 Dockerfile 示例
# ============================================
cat > Dockerfile.training <<'DOCKERFILE'
FROM nvcr.io/nvidia/pytorch:23.10-py3

# 安装 DeepSpeed
RUN pip install --no-cache-dir deepspeed==0.12.4

# 安装 FlashAttention
RUN pip install --no-cache-dir flash-attn==2.3.3

# 安装训练依赖
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    datasets==2.16.0 \
    wandb==0.16.1

# 创建工作目录
WORKDIR /workspace

# 复制训练代码
COPY train.py .
COPY ds_config.json .

# 非 root 用户运行
RUN useradd -m -u 1000 trainer
USER trainer
DOCKERFILE

# 构建并推送
docker build -t registry.maas.local/training:deepspeed-v1.0 .
docker push registry.maas.local/training:deepspeed-v1.0
```

### 7.2 DeepSpeed 配置文件

```bash
# ============================================
# 7.2.1 ds_config.json（ZeRO-3 + Offload）
# ============================================
cat > ds_config.json <<'EOF'
{
  "train_batch_size": 1024,
  "gradient_accumulation_steps": 8,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/mnt/nvme/zero-offload"
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/mnt/nvme/zero-offload"
    },
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  },
  "gradient_clipping": 1.0,
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true
  },
  "wall_clock_breakdown": true,
  "steps_per_print": 10,
  "checkpoint": {
    "use_node_local_storage": true,
    "parallel_write": true
  }
}
EOF
```

### 7.3 多机多卡训练启动

```bash
# ============================================
# 7.3.1 使用 torchrun 启动多机训练
# ============================================
# Master 节点执行
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500
export NNODES=8
export NPROC_PER_NODE=8

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --node_rank=0 \
  train.py \
  --deepspeed \
  --deepspeed_config ds_config.json

# Worker 节点执行（node_rank 从 1 开始递增）
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=<master_ip> \
  --master_port=29500 \
  --node_rank=1 \
  train.py \
  --deepspeed \
  --deepspeed_config ds_config.json

# ============================================
# 7.3.2 Hostfile 方式（DeepSpeed 原生）
# ============================================
# 生成 hostfile
cat > /etc/deepspeed/hostfile <<EOF
train-node-01 slots=8
train-node-02 slots=8
train-node-03 slots=8
train-node-04 slots=8
train-node-05 slots=8
train-node-06 slots=8
train-node-07 slots=8
train-node-08 slots=8
EOF

# 使用 DeepSpeed Launcher
deepspeed --hostfile=/etc/deepspeed/hostfile \
  --master_port=29500 \
  training_script.py \
  --deepspeed \
  --deepspeed_config ds_config.json
```

### 7.4 SSH 免密配置（多机训练必需）

```bash
# ============================================
# 7.4.1 生成 SSH 密钥对
# ============================================
ssh-keygen -t ed25519 -C "training-cluster" -N "" -f ~/.ssh/training_key

# ============================================
# 7.4.2 批量分发公钥
# ============================================
# 使用 pdsh 批量分发（安装 pdsh: apt-get install -y pdsh）
for node in train-node-{01..50}; do
  ssh-copy-id -i ~/.ssh/training_key.pub $node
done

# ============================================
# 7.4.3 添加所有节点到 known_hosts
# ============================================
for node in train-node-{01..50}; do
  ssh-keyscan $node >> ~/.ssh/known_hosts 2>/dev/null
done

# ============================================
# 7.4.4 验证免密登录
# ============================================
ssh -i ~/.ssh/training_key train-node-01 hostname
# 期望输出: train-node-01

# ============================================
# 7.4.5 K8s Secret 方式（推荐）
# ============================================
kubectl create secret generic training-ssh-key \
  --from-file=id_rsa=~/.ssh/training_key \
  --from-file=id_rsa.pub=~/.ssh/training_key.pub \
  -n maas-training
```

---

## 8. 推理体系部署（vLLM + 网关）

### 8.1 vLLM 部署

```bash
# ============================================
# 8.1.1 构建 vLLM 镜像
# ============================================
cat > Dockerfile.vllm <<'DOCKERFILE'
FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN pip install --no-cache-dir vllm==0.3.3
EXPOSE 8000
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
DOCKERFILE

docker build -t registry.maas.local/inference/vllm:v0.3.3 .
docker push registry.maas.local/inference/vllm:v0.3.3

# ============================================
# 8.1.2 部署 vLLM Deployment
# ============================================
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-70b
  namespace: maas-inference
spec:
  replicas: 4
  selector:
    matchLabels:
      app: vllm-llama3-70b
  template:
    metadata:
      labels:
        app: vllm-llama3-70b
    spec:
      nodeSelector:
        node-role: inference
      containers:
        - name: vllm
          image: registry.maas.local/inference/vllm:v0.3.3
          args:
            - "--model"
            - "meta-llama/Llama-3-70B"
            - "--tensor-parallel-size"
            - "4"          # TP=4（4 卡部署 70B 模型）
            - "--max-model-len"
            - "8192"
            - "--gpu-memory-utilization"
            - "0.95"
            - "--enable-prefix-caching"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 4
              memory: "128Gi"
              cpu: "32"
            requests:
              nvidia.com/gpu: 4
              memory: "128Gi"
              cpu: "32"
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 300   # 模型加载需要时间
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 300
            periodSeconds: 10
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: vllm-model-cache
EOF

# ============================================
# 8.1.3 创建 Service
# ============================================
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama3-70b
  namespace: maas-inference
spec:
  selector:
    app: vllm-llama3-70b
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
EOF
```

### 8.2 推理网关（Kong / Envoy）

```bash
# ============================================
# 8.2.1 部署 Kong 网关
# ============================================
helm repo add kong https://charts.konghq.com
helm repo update

helm install kong kong/ingress \
  --namespace kong \
  --create-namespace \
  --set ingressController.installCRDs=false \
  --set admin.enabled=false \
  --set proxy.type=LoadBalancer

# ============================================
# 8.2.2 配置路由规则
# ============================================
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: inference-ingress
  namespace: maas-inference
  annotations:
    konghq.com/strip-path: "true"
    konghq.com/plugins: rate-limit
spec:
  ingressClassName: kong
  rules:
    - host: inference.maas.local
      http:
        paths:
          - path: /v1/llama3-70b
            pathType: Prefix
            backend:
              service:
                name: vllm-llama3-70b
                port:
                  number: 8000
          - path: /v1/qwen2-72b
            pathType: Prefix
            backend:
              service:
                name: vllm-qwen2-72b
                port:
                  number: 8000
EOF
```

### 8.3 推理 HPA（自动扩缩容）

```bash
# ============================================
# 8.3.1 基于排队长度的 HPA
# ============================================
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: maas-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama3-70b
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: inference_queue_length
        target:
          type: AverageValue
          averageValue: "5"     # 平均排队 > 5 时扩容
    - type: Pods
      pods:
        metric:
          name: inference_p99_latency
        target:
          type: AverageValue
          averageValue: "200"   # P99 延迟 > 200ms 时扩容
EOF
```

---

## 9. 可观测性（Prometheus + DCGM + Grafana + Loki）

### 9.1 kube-prometheus-stack 一键部署

```bash
# ============================================
# 9.1.1 添加 Prometheus Helm 仓库
# ============================================
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# ============================================
# 9.1.2 部署全套监控栈
# ============================================
kubectl create namespace monitoring

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --version 56.10.0 \
  --set prometheus.prometheusSpec.retention=7d \
  --set prometheus.prometheusSpec.resources.requests.memory=4Gi \
  --set grafana.adminPassword=grafana-admin \
  --set alertmanager.alertmanagerSpec.replicas=2

# 验证
kubectl get pods -n monitoring
# 期望看到:
#   prometheus-kube-prometheus-stack-prometheus-0   2/2 Running
#   prometheus-grafana-xxx                          3/3 Running
#   prometheus-alertmanager-xxx                     2/2 Running
```

### 9.2 DCGM Exporter 部署

```bash
# ============================================
# 9.2.1 GPU Operator 已包含 DCGM Exporter
# ============================================
# 验证 DCGM Exporter 正常运行
kubectl get pods -n gpu-operator -l app=nvidia-dcgm-exporter

# 验证指标采集
kubectl port-forward -n gpu-operator \
  $(kubectl get pods -n gpu-operator -l app=nvidia-dcgm-exporter -o jsonpath='{.items[0].metadata.name}') \
  9400:9400 &

curl http://localhost:9400/metrics | head -30
# 期望看到 DCGM 指标:
#   DCGM_FI_DEV_GPU_TEMP{gpu="0"} 65
#   DCGM_FI_DEV_POWER_USAGE{gpu="0"} 250
#   DCGM_FI_DEV_FB_USED{gpu="0"} 40000
#   DCGM_FI_DEV_ECC_DBE_VOL{gpu="0"} 0

# ============================================
# 9.2.2 配置 Prometheus ServiceMonitor（自动发现 DCGM）
# ============================================
# GPU Operator 默认创建 ServiceMonitor
kubectl get servicemonitor -n gpu-operator
# 期望输出: dcgm-exporter
```

### 9.3 Grafana Dashboard 导入

```bash
# ============================================
# 9.3.1 访问 Grafana
# ============================================
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# 浏览器打开: http://localhost:3000
# 默认账号: admin / grafana-admin

# ============================================
# 9.3.2 导入预置 Dashboard
# ============================================
# DCGM Exporter Dashboard (ID: 12239)
# Kubernetes Cluster Monitoring (ID: 7249)
# NVIDIA GPU Dashboard (ID: 9965)

# 通过 Grafana CLI 导入
# 或在 Grafana UI 中: + → Import → 输入 Dashboard ID

# ============================================
# 9.3.3 自定义 GPU 监控 Dashboard JSON
# ============================================
# 关键 Panel：
# 1. GPU 温度热力图（按节点）
# 2. GPU 功耗实时曲线
# 3. 显存使用率（按 Pod 分组）
# 4. GPU 利用率（训练 vs 空闲）
# 5. NVLink 带宽利用率
# 6. ECC 错误计数趋势
# 7. XID 错误实时告警
```

### 9.4 Loki 日志聚合

```bash
# ============================================
# 9.4.1 部署 Loki Stack
# ============================================
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set promtail.enabled=true \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=100Gi \
  --set loki.persistence.storageClass=ceph-rbd-nvme

# ============================================
# 9.4.2 配置 Promtail 采集训练日志
# ============================================
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-extra-config
  namespace: monitoring
data:
  extra-config.yaml: |
    snippets:
      extraScrapeConfigs: |
        - job_name: training-logs
          kubernetes_sd_configs:
            - role: pod
          relabel_configs:
            - source_labels: [__meta_kubernetes_pod_label_app]
              regex: training-job
              action: keep
            - source_labels: [__meta_kubernetes_pod_name]
              target_label: pod
            - source_labels: [__meta_kubernetes_namespace]
              target_label: namespace
EOF
```

### 9.5 告警规则配置

```bash
# ============================================
# 9.5.1 Prometheus 告警规则
# ============================================
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: maas-gpu-alerts
  namespace: monitoring
  labels:
    app: kube-prometheus-stack
    release: prometheus
spec:
  groups:
    - name: gpu-hardware.rules
      rules:
        # P0: GPU 温度过高
        - alert: GpuTemperatureCritical
          expr: DCGM_FI_DEV_GPU_TEMP > 87
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "GPU {{ \$labels.gpu }} 温度 {{ \$value }}°C 超过临界值"

        # P1: GPU ECC 双比特错误
        - alert: GpuEccDoubleBitError
          expr: DCGM_FI_DEV_ECC_DBE_VOL > 0
          for: 0m
          labels:
            severity: warning
          annotations:
            summary: "GPU {{ \$labels.gpu }} 检测到 ECC 双比特错误"

        # P1: GPU XID 错误
        - alert: GpuXidError
          expr: DCGM_FI_DEV_XID_ERRORS > 0
          for: 0m
          labels:
            severity: warning
          annotations:
            summary: "GPU {{ \$labels.gpu }} XID 错误码 {{ \$value }}"

        # P2: GPU 显存使用率过高
        - alert: GpuMemoryHigh
          expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE > 0.95
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "GPU {{ \$labels.gpu }} 显存使用率 {{ \$value | humanizePercentage }}"

    - name: k8s-stability.rules
      rules:
        # P0: 节点 NotReady
        - alert: K8sNodeNotReady
          expr: kube_node_status_condition{condition="Ready",status="true"} == 0
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "节点 {{ \$labels.node }} NotReady 超过 5 分钟"

        # P1: Pod 反复重启
        - alert: K8sPodCrashLooping
          expr: rate(kube_pod_container_status_restarts_total[15m]) * 60 * 15 > 3
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Pod {{ \$labels.pod }} 在 15 分钟内重启超过 3 次"
EOF
```

---

## 10. 安全加固（RBAC + 镜像扫描 + SSH 密钥）

### 10.1 RBAC 角色定义

```bash
# ============================================
# 10.1.1 训练操作员角色
# ============================================
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: training-operator
rules:
  - apiGroups: ["batch.volcano.sh"]
    resources: ["jobs", "podgroups"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: training-operator-binding
subjects:
  - kind: Group
    name: training-team
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: training-operator
  apiGroup: rbac.authorization.k8s.io
EOF

# ============================================
# 10.1.2 Namespace 资源配额
# ============================================
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: maas-training-quota
  namespace: maas-training
spec:
  hard:
    requests.nvidia.com/gpu: "64"
    limits.nvidia.com/gpu: "64"
    memory: "4Ti"
    cpu: "1024"
    persistentvolumeclaims: "20"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: maas-training-limits
  namespace: maas-training
spec:
  limits:
    - type: Container
      default:
        nvidia.com/gpu: "1"
        memory: "32Gi"
        cpu: "8"
      defaultRequest:
        nvidia.com/gpu: "1"
        memory: "32Gi"
        cpu: "8"
EOF
```

### 10.2 镜像安全扫描（Trivy）

```bash
# ============================================
# 10.2.1 安装 Trivy Operator
# ============================================
helm repo add aqua https://aquasecurity.github.io/helm-charts
helm repo update

helm install trivy-operator aqua/trivy-operator \
  --namespace trivy-system \
  --create-namespace \
  --set trivy.ignoreUnfixed=true

# ============================================
# 10.2.2 扫描训练镜像
# ============================================
# 本地扫描
trivy image --severity CRITICAL,HIGH registry.maas.local/training:deepspeed-v1.0

# CI/CD 流水线集成
# 在构建后自动运行 Trivy，发现 CRITICAL 级别漏洞时阻断推送

# ============================================
# 10.2.3 镜像签名（Cosign）
# ============================================
# 生成密钥对
cosign generate-key-pair

# 签名镜像
cosign sign --key cosign.key registry.maas.local/training:deepspeed-v1.0

# 验证签名
cosign verify --key cosign.pub registry.maas.local/training:deepspeed-v1.0
```

### 10.3 SSH 密钥管理（K8s Secret）

```bash
# ============================================
# 10.3.1 创建 SSH 密钥 Secret
# ============================================
kubectl create secret generic training-ssh-keys \
  --from-file=ssh-privatekey=~/.ssh/training_key \
  --from-file=ssh-publickey=~/.ssh/training_key.pub \
  --from-file=authorized_keys=~/.ssh/training_key.pub \
  -n maas-training

# ============================================
# 10.3.2 挂载到训练 Pod
# ============================================
# 在训练 Pod spec 中添加:
# volumeMounts:
#   - name: ssh-keys
#     mountPath: /root/.ssh
#     readOnly: true
# volumes:
#   - name: ssh-keys
#     secret:
#       secretName: training-ssh-keys
#       defaultMode: 0600

# ============================================
# 10.3.3 密钥轮换（CronJob）
# ============================================
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: ssh-key-rotation
  namespace: maas-training
spec:
  schedule: "0 0 1 * *"    # 每月 1 号轮换
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: rotate
              image: bitnami/kubectl:latest
              command:
                - bash
                - -c
                - |
                  ssh-keygen -t ed25519 -C "training-cluster" -N "" -f /tmp/key
                  kubectl delete secret training-ssh-keys -n maas-training
                  kubectl create secret generic training-ssh-keys \
                    --from-file=ssh-privatekey=/tmp/key \
                    --from-file=ssh-publickey=/tmp/key.pub \
                    --from-file=authorized_keys=/tmp/key.pub \
                    -n maas-training
          restartPolicy: OnFailure
EOF
```

---

## 11. 日常运维 Command 速查

### 11.1 GPU 健康检查

```bash
# 快速检查所有 GPU 节点健康状态
for node in $(kubectl get nodes -l node-role=training -o jsonpath='{.items[*].metadata.name}'); do
  echo "=== $node ==="
  kubectl exec -n gpu-operator \
    $(kubectl get pods -n gpu-operator -l app=nvidia-device-plugin-daemonset \
      --field-selector spec.nodeName=$node -o jsonpath='{.items[0].metadata.name}') \
    -- nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
done

# DCGM 快速诊断
kubectl -n gpu-operator exec -it \
  $(kubectl get pods -n gpu-operator -l app=nvidia-dcgm-exporter -o jsonpath='{.items[0].metadata.name}') \
  -- dcgmi diag -r 1   # 快速诊断
```

### 11.2 NCCL 调试

```bash
# 开启 NCCL 调试日志（训练前设置环境变量）
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET,COLL

# 检查 P2P 是否可用
nvidia-smi topo -m
# GPU0 GPU1 GPU2 ...  CPU Affinity  NUMA Affinity
# GPU0   X   NV8 NV8 ...  0-15          N/A
# GPU1  NV8   X  NV8 ...  0-15          N/A
# "NV8" = NVLink 连接，"PIX" = PCIe 同交换机，"PHB" = PCIe 同 CPU

# 测试 NCCL 性能
cd nccl-tests
./build/all_reduce_perf -b 8 -e 4G -f 2 -g 8 -c 1
```

### 11.3 训练任务管理

```bash
# 查看所有训练任务
kubectl get jobs -n maas-training

# 查看训练任务日志
kubectl logs -f -n maas-training -l app=training-job --tail=100

# 查看特定 Rank 的日志
kubectl logs -f -n maas-training training-job-001-0 -c trainer

# 暂停训练（Cordon + Drain）
kubectl cordon train-node-05
kubectl drain train-node-05 --ignore-daemonsets --delete-emptydir-data

# 恢复训练节点
kubectl uncordon train-node-05

# 强制删除卡住的 Pod
kubectl delete pod <pod-name> -n maas-training --grace-period=0 --force
```

### 11.4 存储运维

```bash
# 查看 Ceph 集群状态
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- ceph status
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- ceph osd status
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- ceph df

# 查看 PVC 使用情况
kubectl get pvc -A

# Ceph 限流设置（扩容时）
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- \
  ceph config set osd osd_max_backfills 1

# 查看存储使用率
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- \
  ceph df detail
```

### 11.5 网络诊断

```bash
# 检查 IB 网络状态
ibstat
ibv_devinfo
ibnetdiscover

# RDMA 带宽测试
# Server
ib_write_bw -d mlx5_0
# Client
ib_write_bw -d mlx5_0 <server_ip>

# MTU 测试
ping -M do -s 4064 <target_ib_ip>    # IB: MTU 4092
ping -M do -s 8972 <target_stor_ip>  # 存储: MTU 9000

# 网络延迟测试
mtr -n <target_ip>

# Pod 间网络连通性测试
kubectl exec -it <pod-a> -- nc -zv <pod-b-ip> 23456
```

### 11.6 集群备份

```bash
# ============================================
# etcd 备份
# ============================================
ETCDCTL_API=3 etcdctl snapshot save /backup/etcd-snapshot-$(date +%Y%m%d-%H%M%S).db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

# ============================================
# K8s 资源清单备份
# ============================================
kubectl get all -A -o yaml > /backup/k8s-all-resources-$(date +%Y%m%d).yaml

# ============================================
# Helm Release 备份
# ============================================
helm list -A -o yaml > /backup/helm-releases-$(date +%Y%m%d).yaml
```

---

## 附录：完整部署顺序检查清单

```
□ 阶段一：PoC（5-10 节点）
  □ 1. 操作系统安装 + 内核/驱动基线
  □ 2. containerd + kubeadm/kubelet/kubectl 安装
  □ 3. kubeadm init（Master 初始化）
  □ 4. Worker 节点 join
  □ 5. Calico CNI 部署（BGP 模式）
  □ 6. GPU Operator 部署
  □ 7. GPU 资源注册验证（nvidia-smi in Pod）
  □ 8. IB/RoCE 驱动安装 + RDMA 验证
  □ 9. GPUDirect RDMA 配置 + 验证
  □ 10. MTU 一致性检查
  □ 11. NCCL 基准测试（单机 8 卡 AllReduce）
  □ 12. DeepSpeed 小模型训练验证

□ 阶段二：小规模（20-30 节点）
  □ 13. Ceph 存储集群部署（Rook）
  □ 14. CephFS + RBD 存储池创建
  □ 15. MinIO 对象存储部署（L3）
  □ 16. 三级缓存链路验证
  □ 17. Volcano 调度器部署
  □ 18. Gang Scheduling + Binpack 验证
  □ 19. NetworkPolicy 配置
  □ 20. Prometheus + DCGM + Grafana 部署
  □ 21. Loki 日志聚合部署
  □ 22. 端到端训练任务跑通（8→64 卡线性比 ≥ 85%）

□ 阶段三：全量部署（100 节点）
  □ 23. SuperPOD 网络部署（Spine-Leaf IB）
  □ 24. 全量 K8s 节点加入
  □ 25. GPU Operator 全量部署
  □ 26. 多机多卡训练（100 节点，800 卡）
  □ 27. vLLM 推理服务部署
  □ 28. 压测 + MFU 调优（目标 ≥ 40%）
  □ 29. 故障注入测试（Kill 1 节点自动恢复）
  □ 30. RBAC + 镜像安全 + SSH 密钥加固
  □ 31. 告警规则配置 + 通知通道验证
  □ 32. etcd + 资源清单备份策略
  □ 33. 生产验收
```

---

> **说明**：本手册基于 maas-platform-full.md 架构文档编写，将架构设计落地为可执行的 Command 命令集。所有命令均经过逻辑验证，但实际部署时需根据硬件型号、网络拓扑和软件版本进行适配。建议严格按照附录中的检查清单逐步执行，不要跳过任何验证步骤。
