def get_available_cpus():
    import os
    import multiprocessing
    """
    获取可用的CPU核数，优先使用SLURM环境变量，否则使用系统CPU数

    Returns:
        int: 可用的CPU核数
    """
    # 尝试从SLURM环境变量获取
    slurm_cpus = os.environ.get('SLURM_CPUS_ON_NODE')

    if slurm_cpus is not None:
        try:
            # 处理可能的格式（可能是逗号分隔的列表或单个数字）
            if ',' in slurm_cpus:
                # 如果是逗号分隔的列表，计算总数
                cpus_list = slurm_cpus.split(',')
                total_cpus = sum(int(cpu) for cpu in cpus_list if cpu.strip().isdigit())
            else:
                # 如果是单个数字
                total_cpus = int(slurm_cpus)

            # 确保至少有一个CPU可用
            return max(1, total_cpus)

        except (ValueError, TypeError):
                # 如果转换失败，回退到系统CPU数
                pass

    # 如果没有SLURM环境变量或转换失败，使用系统CPU数
    try:
        system_cpus = multiprocessing.cpu_count()
        return max(1, system_cpus)
    except NotImplementedError:
        # 如果无法获取系统CPU数，返回默认值1
        return 1
