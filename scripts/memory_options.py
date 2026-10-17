"""命令行共用的设备与显存预算参数。"""


def add_memory_arguments(parser):
    parser.add_argument("--device", type=int, default=0, help="进程可见的加速卡索引")
    parser.add_argument("--memory-budget-mib", type=int, default=0, help="引擎显存预算，0 仅按整卡比例规划")
    parser.add_argument("--device-reserve-mib", type=int, default=0, help="为其他模型及工作区预留的整卡显存")
    parser.add_argument("--memory-workspace-mib", type=int, default=256, help="运行工作区最低预留")
    parser.add_argument("--graph-reserve-mib", type=int, default=256, help="Graph 额外预留；eager 不使用")


def memory_kwargs(args):
    return dict(device_ids=(args.device,), memory_budget_bytes=args.memory_budget_mib * 2**20,
                device_memory_reserve_bytes=args.device_reserve_mib * 2**20,
                memory_workspace_bytes=args.memory_workspace_mib * 2**20,
                graph_memory_reserve_bytes=args.graph_reserve_mib * 2**20)
