import sys
import os, re
import time
from bisect import bisect_left
from multiprocessing import Pool, cpu_count


def load_mapping(mapping_file):
    mnemonic_to_group = {}
    groups = set()
    with open(mapping_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            mnemonic = parts[1]
            group_name = parts[2]
            mnemonic_to_group[mnemonic] = group_name
            groups.add(group_name)
    return mnemonic_to_group, sorted(list(groups))


def load_reference(ref_bed):
    annotations = {}
    all_states = set()
    print(f"[*] Reading reference file (this may take ~30s)...")
    start_time = time.time()
    with open(ref_bed, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            chrom, start, end, state_full = parts[0], int(parts[1]), int(parts[2]), parts[3]
            if chrom not in annotations:
                annotations[chrom] = []
            annotations[chrom].append((start, end, state_full))
            all_states.add(state_full)

    for chrom in annotations:
        annotations[chrom].sort()

    sorted_states = sorted(list(all_states), key=lambda x: (int(x.split('_')[0]), x))
    print(f"[*] Reference loaded in {time.time() - start_time:.2f}s.")
    return annotations, sorted_states


def calculate_overlap(query_chrom, query_start, query_end, chrom_ref):
    ref_chrom = query_chrom if query_chrom.startswith('chr') else f"chr{query_chrom}"
    if ref_chrom not in chrom_ref:
        return {}

    ref_list = chrom_ref[ref_chrom]
    starts = [x[0] for x in ref_list]
    idx = bisect_left(starts, query_start)
    if idx > 0:
        idx -= 1

    overlaps = {}
    for i in range(idx, len(ref_list)):
        ref_start, ref_end, state_full = ref_list[i]
        if ref_start >= query_end:
            break
        overlap_s = max(query_start, ref_start)
        overlap_e = min(query_end, ref_end)
        if overlap_s < overlap_e:
            overlaps[state_full] = overlaps.get(state_full, 0) + (overlap_e - overlap_s)
    return overlaps


def worker_task(query_file, annotations, all_states, all_groups, mnemonic_to_group, output_dir):
    prefix = os.path.basename(query_file).replace('.bed.out', '').replace('.bed', '')
    state_results = []
    group_results = []
    region_pattern = re.compile(r'^([^:]+):(\d+)-(\d+)$')

    with open(query_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or "region" in line.lower() or line.startswith("#"):
                continue
            m = region_pattern.match(line)
            if m:
                chrom, start, end = m.groups()
                parts = [chrom, start, end]
            else:
                parts = line.split()
                if len(parts) < 3:
                    continue

            chrom, q_start, q_end = parts[0], int(parts[1]), int(parts[2])
            region_id = f"chr{chrom}:{q_start}-{q_end}" if not chrom.startswith('chr') else f"{chrom}:{q_start}-{q_end}"
            total_len = q_end - q_start
            if total_len <= 0: continue

            overlap_counts = calculate_overlap(chrom, q_start, q_end, annotations)

            s_p = [f"{(overlap_counts.get(s, 0) / total_len * 100):.4f}" for s in all_states]
            state_results.append(f"{region_id}\t" + "\t".join(s_p))

            group_counts = {g: 0 for g in all_groups}
            for s_full, length in overlap_counts.items():
                mnemonic_name = s_full.split('_')[-1]
                g = mnemonic_to_group.get(mnemonic_name)
                if g:
                    group_counts[g] += length

            g_p = [f"{(group_counts[g] / total_len * 100):.4f}" for g in all_groups]
            group_results.append(f"{region_id}\t" + "\t".join(g_p))

    # 写入结果
    with open(os.path.join(output_dir, f"{prefix}.output.txt"), 'w') as f:
        f.write("Region\t" + "\t".join(all_states) + "\n")
        f.write("\n".join(state_results) + "\n")

    with open(os.path.join(output_dir, f"{prefix}.grouped_output.txt"), 'w') as f:
        f.write("Region\t" + "\t".join(all_groups) + "\n")
        f.write("\n".join(group_results) + "\n")

    return f"Finished {prefix}"


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 chrom_full_stack_parallel.py <ref_dir> <output_dir> <input1.bed> [input2.bed ...]")
        sys.exit(1)

    ref_dir = sys.argv[1]
    output_dir = sys.argv[2]
    input_files = sys.argv[3:]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ref_bed = os.path.join(ref_dir, 'hg38lift_genome_100_segments.bed')
    mapping_file = os.path.join(ref_dir, 'state_annotations_processed')

    mnemonic_to_group, all_groups = load_mapping(mapping_file)
    annotations, all_states = load_reference(ref_bed)

    num_cores = min(8, cpu_count(), len(input_files))
    print(f"[*] Parallel mode: {num_cores} cores. Files: {len(input_files)}")

    tasks = [(f, annotations, all_states, all_groups, mnemonic_to_group, output_dir) for f in input_files]

    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(worker_task, tasks)

    for res in results:
        print(f"  {res}")

    print(f"[*] All tasks completed in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()