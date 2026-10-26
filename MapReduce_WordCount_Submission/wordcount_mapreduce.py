"""Small local MapReduce word-count engine (no Hadoop or Spark).

Usage:
    Run this file in IDLE or another Python editor for folder-selection windows.
    Or: python wordcount_mapreduce.py INPUT_FOLDER OUTPUT_FOLDER

Each .txt file directly inside INPUT_FOLDER is one input split. The map tasks
run in separate processes; there is one task per split. Two reducer processes
produce part-00000.txt and part-00001.txt.
"""

import argparse
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def map_words(text):
    """Map: emit (word, 1) for every word in a split."""
    return [(word, 1) for word in re.findall(r"[a-z0-9]+", text.lower())]


def combine(pairs):
    """Combine: sum repeated words within just this mapper's output."""
    local_totals = defaultdict(int)
    for word, count in pairs:
        local_totals[word] += count
    return list(local_totals.items())


def partition(word):
    """Send the same word to the same reducer every time."""
    # Deterministic two-way partition: a-m and digits -> 0; n-z -> 1.
    return 1 if "n" <= word[0] <= "z" else 0


def map_combine_partition(file_path):
    """One worker performs all three map-side phases for one input file."""
    text = Path(file_path).read_text(encoding="utf-8")
    mapped = map_words(text)
    combined = combine(mapped)
    partitions = [[], []]
    for word, subtotal in combined:
        partitions[partition(word)].append((word, subtotal))
    return partitions


def shuffle_and_sort(mapper_results):
    """Merge mapper partitions; sort and group values for each reducer."""
    incoming = [[], []]
    for mapper_partitions in mapper_results:
        for reducer_id in range(2):
            incoming[reducer_id].extend(mapper_partitions[reducer_id])

    grouped = []
    for reducer_pairs in incoming:
        values_by_word = defaultdict(list)
        for word, subtotal in sorted(reducer_pairs):
            values_by_word[word].append(subtotal)
        grouped.append(sorted(values_by_word.items()))
    return grouped


def reduce_and_write(job):
    """Reduce: sum all subtotals per word and write one reducer file."""
    reducer_id, grouped_pairs, output_dir = job
    output_path = Path(output_dir) / f"part-{reducer_id:05d}.txt"
    with output_path.open("w", encoding="utf-8") as output:
        for word, values in grouped_pairs:
            output.write(f"{word}\t{sum(values)}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Local MapReduce word count")
    parser.add_argument("input_folder", nargs="?", type=Path)
    parser.add_argument("output_folder", nargs="?", type=Path)
    args = parser.parse_args()

    gui_mode = args.input_folder is None and args.output_folder is None
    if gui_mode:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.withdraw()
        input_choice = filedialog.askdirectory(title="Choose the folder containing .txt files")
        if not input_choice:
            return
        output_choice = filedialog.askdirectory(title="Choose a separate output folder")
        if not output_choice:
            return
        args.input_folder = Path(input_choice)
        args.output_folder = Path(output_choice)
    elif args.input_folder is None or args.output_folder is None:
        parser.error("Provide both folders, or run without arguments for folder selection")

    input_folder = args.input_folder.resolve()
    output_folder = args.output_folder.resolve()
    if not input_folder.is_dir() or input_folder == output_folder:
        error = "Choose an existing input folder and a different output folder"
        if gui_mode:
            messagebox.showerror("Word Count", error)
            return
        parser.error(error)

    files = sorted(input_folder.glob("*.txt"))
    if not files:
        error = "No .txt files found directly inside the input folder"
        if gui_mode:
            messagebox.showerror("Word Count", error)
            return
        parser.error(error)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Submit all mappers to separate worker processes at the same time.
    with ProcessPoolExecutor(max_workers=len(files)) as pool:
        futures = [pool.submit(map_combine_partition, path) for path in files]
        mapper_results = [future.result() for future in futures]

    reducer_inputs = shuffle_and_sort(mapper_results)

    # Run the two reducers in parallel, writing one file per reducer.
    jobs = [(i, reducer_inputs[i], output_folder) for i in range(2)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        output_paths = list(pool.map(reduce_and_write, jobs))

    print(f"Processed {len(files)} input split(s).")
    for path in output_paths:
        print(f"Wrote {path}")
    if gui_mode:
        messagebox.showinfo(
            "Word Count Complete",
            f"Processed {len(files)} text file(s).\n\nResults saved in:\n{output_folder}\n\n"
            "Open part-00000.txt and part-00001.txt to see the counts.",
        )
        root.destroy()


if __name__ == "__main__":
    main()
