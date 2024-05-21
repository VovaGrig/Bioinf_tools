from time import time
from dataclasses import dataclass
from typing import List
import os

# HW15


class MeasureTime:
    """
    Context manager for measuring the execution time of a code block.

    Example
    -------
    with MeasureTime():
        # Code block to measure
        pass

    The elapsed time will be printed when the code block exits.
    """

    def __init__(self):
        self.start_time = None

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(
            "The program block took %s seconds to complete" % (time() - self.start_time)
        )


class OpenFasta:
    """
    Context manager and iterator for reading records from a FASTA file.

    This class provides an easy way to read and iterate over records in a FASTA file.
    It supports usage as a context manager to ensure the file is properly opened and closed.

    Parameters
    ----------
    file_path : str
        The path to the FASTA file to be read.

    Attributes
    ----------
    file_path : str
        The path to the FASTA file.
    current_line : str
        The current line being processed from the file.
    handler : file object
        The file handler for the FASTA file.

    Methods
    -------
    __enter__()
        Opens the FASTA file and returns the iterator object.
    __next__()
        Reads the next FASTA record from the file.
    read_record()
        Reads the next FASTA record using the __next__ method.
    read_records()
        Reads all the FASTA records from the file.
    __iter__()
        Returns the iterator object.
    __exit__(exc_type, exc_val, exc_tb)
        Closes the FASTA file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.current_line = None

    def __enter__(self):
        self.handler = open(self.file_path)
        return self

    def __next__(self):
        if self.current_line != None:
            current_line = self.current_line
        else:
            current_line = self.handler.readline().strip()
        if current_line == "":
            raise StopIteration("No more records in provided fasta file.")
        if current_line.startswith(">"):
            id = current_line.split(" ")[0]
            description = " ".join(current_line.split(" ")[1:])
            current_seq = ""
            current_line = self.handler.readline().strip()
            while not (current_line.startswith(">") or current_line == ""):
                current_seq += current_line.strip()
                current_line = self.handler.readline().strip()
            record = FastaRecord(id, current_seq.rstrip("\n"), description)
        self.current_line = current_line
        return record

    def read_record(self):
        return self.__next__()

    def read_records(self):
        return list(self.__iter__())

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handler.close()


@dataclass
class FastaRecord:
    """
    Data class for storing information about a single FASTA record.

    Parameters
    ----------
    id : str
        The identifier of the FASTA record.

    seq : str
        The sequence of the FASTA record.

    description : str
        The description of the FASTA record.

    Methods
    -------
    __repr__()
        Returns a string representation of the FASTA record, including the ID, description, and sequence.
    """

    id: str
    seq: str
    description: str

    def __repr__(self):
        return (
            f"ID: {self.id},\n Description: {self.description},\n Sequence:{self.seq}\n"
        )


# HW17


@dataclass
class GenscanOutput:
    """
    Data class for storing the output of the GENSCAN prediction.

    Parameters
    ----------
    status : int
        The status code of the GENSCAN service response.

    cds_list : List[dict]
        A list of dictionaries, each containing information about a predicted CDS (coding sequence).

    intron_list : List[dict]
        A list of dictionaries, each containing information about a predicted intron.

    exon_list : List[dict]
        A list of dictionaries, each containing information about a predicted exon.
    """

    status: int
    cds_list: List[dict]
    intron_list: List[dict]
    exon_list: List[dict]


def extract_exons(lines):
    """
    Extracts exon information from the GENSCAN output lines.

    Parameters
    ----------
    lines : list of str
        The lines from the GENSCAN output.

    Returns
    -------
    list of dict
        A list of dictionaries containing information about each predicted exon.
    """
    exon_list = []
    exon_start_section = lines.index(
        "Gn.Ex Type S .Begin ...End .Len Fr Ph I/Ac Do/T CodRg P.... Tscr.."
    )
    exon_end_section = lines.index("Suboptimal exons with probability > 1.000")
    exon_lines = lines[exon_start_section + 6 : exon_end_section - 5]
    for line in exon_lines:
        if not line:
            continue
        parts = line.split()
        number = parts[0]
        type = parts[1]
        start = int(parts[3])
        end = int(parts[4])
        exon_info = {"number": number, "type": type, "start": start, "end": end}
        exon_list.append(exon_info)
    return exon_list


def extract_introns(lines):
    """
    Extracts intron information from the GENSCAN output lines.

    Parameters
    ----------
    lines : list of str
        The lines from the GENSCAN output.

    Returns
    -------
    list of dict
        A list of dictionaries containing information about each predicted intron.
    """
    exon_list = extract_exons(lines)
    sequence_length = [x for x in lines if x.startswith("Sequence")][0]
    sequence_length = sequence_length.split(" : ")[1]
    sequence_length = int(sequence_length.split()[0])
    intron_list = []
    for i in range(len(exon_list) - 1):
        current_exon = exon_list[i]
        next_exon = exon_list[i + 1]

        intron_start = None
        intron_end = None

        if i == 0 and int(current_exon["start"]) > 1:
            intron_start = 1
            intron_end = current_exon["start"] - 1
            append_to_intron_list(
                intron_list, "init", intron_start, intron_end, current_exon, next_exon
            )
        intron_start = current_exon["end"] + 1
        intron_end = next_exon["start"] - 1
        append_to_intron_list(
            intron_list, "intr", intron_start, intron_end, current_exon, next_exon
        )
        if i == (len(exon_list) - 2) and int(next_exon["end"]) < sequence_length:
            intron_start = next_exon["end"] + 1
            intron_end = sequence_length
            append_to_intron_list(
                intron_list, "term", intron_start, intron_end, current_exon, next_exon
            )
    return intron_list


def extract_predicted_peptides(lines):
    """
    Extracts predicted peptide information from the GENSCAN output lines.

    Parameters
    ----------
    lines : list of str
        The lines from the GENSCAN output.

    Returns
    -------
    list of dict
        A list of dictionaries containing information about each predicted peptide.
    """
    cds_list = []
    cds_start_section = lines.index("Predicted peptide sequence(s):")
    cds_lines = lines[cds_start_section + 6 :]
    cds_seq = []
    line_split = None
    peptide_info = None
    for count, line in enumerate(cds_lines):
        header = None
        if line.startswith(">"):
            header = True
            if count != 0:
                cds_list.append(
                    {
                        "peptide": peptide_info,
                        "sequence": "".join(cds_seq),
                    }
                )
            line_split = line.split("|")
            peptide_info = line_split[1:]
        if not header and line:
            cds_seq.append(line)
        if count == len(cds_lines) - 1:
            cds_list.append(
                {
                    "peptide": peptide_info,
                    "sequence": "".join(cds_seq),
                }
            )
    return cds_list


def append_to_intron_list(intron_list, type, start, end, current_exon, next_exon):
    """
    Appends intron information to the intron list.

    Parameters
    ----------
    intron_list : list of dict
        The list to append intron information to.

    type : str
        The type of the intron (init, intr, term).

    start : int
        The start position of the intron.

    end : int
        The end position of the intron.

    current_exon : dict
        The current exon information.

    next_exon : dict
        The next exon information.
    """
    number_dict = {
        "init": f"1 - {current_exon['number']}",
        "intr": f"{current_exon['number']}-{next_exon['number']}",
        "term": f"{next_exon['number']}-end",
    }
    intron_list.append(
        {
            "number": number_dict[type],
            "type": "Intron",
            "start": start,
            "end": end,
        }
    )


# HW6


def convert_multiline_fasta_to_oneline(input_fasta: str, output_fasta: str = ""):
    """
    Converts a multi-line FASTA file to a single-line format for each sequence.

    Parameters
    ----------
    input_fasta : str
        Path to the input multi-line FASTA file.

    output_fasta : str, optional
        Path to the output single-line FASTA file. If not provided, the output file will be named
        based on the input file with a suffix '_oneline.fasta'.
    """
    if output_fasta == "":
        output_fasta = os.path.basename(input_fasta)
        output_fasta = output_fasta.replace(".fasta", "_oneline.fasta")
    output_path = os.path.join("./", output_fasta)
    with open(input_fasta, "r") as input:
        with open(output_path, "w"):
            read = []
            while True:
                line = input.readline().strip()
                if not line:
                    break
                if line.startswith(">"):
                    line += "\n"
                    if read:
                        with open(output_path, "a") as output:
                            output.write("".join(read) + "\n")
                    read = [line]
                else:
                    read.append(line)
            with open(output_path, "a") as output:
                output.write("".join(read))


def select_genes_from_gbk_to_fasta(
    input_gbk: str,
    genes: str or tuple[str] or list[str],
    n_before: int = 1,
    n_after: int = 1,
    output_fasta: str = "",
):
    """
    Selects specified genes from a GenBank file and writes their translations to a FASTA file.

    Parameters
    ----------
    input_gbk : str
        Path to the input GenBank file.

    genes : str, tuple of str, or list of str
        Genes to be selected from the GenBank file.

    n_before : int, default=1
        Number of genes before the specified gene to include in the output.

    n_after : int, default=1
        Number of genes after the specified gene to include in the output.

    output_fasta : str, optional
        Path to the output FASTA file. If not provided, the output file will be named
        based on the input file with a suffix '_trans_for_blast.fasta'.
    """
    if output_fasta == "":
        output_fasta = os.path.basename(input_gbk)
        output_fasta = output_fasta.replace(".gbk", "_trans_for_blast.fasta")
    output_path = os.path.join("./", output_fasta)
    if isinstance(genes, str):
        genes = [genes]
    with open(input_gbk, "r") as input:
        translations = []
        ind = 0
        prev_gene = "undefined"
        was_printed = set()
        while True:
            line = input.readline().strip()
            if not line:
                break
            if line.startswith("/gene="):
                prev_gene = line[7:-1]
            if line.startswith("/translation="):
                translation = line[14:]
                while not translation.endswith('"'):
                    line = input.readline().strip()
                    translation += line
                translation = translation.rstrip('"')
                translations.append((prev_gene, translation))
                prev_gene = "undefined"
        with open(output_path, "w") as out:
            for gene in genes:
                for ind, cur in enumerate(translations):
                    if gene in cur[0]:
                        for i in range(
                            max(0, ind - n_before),
                            min(len(translations), ind + n_after + 1),
                        ):
                            if i == ind or i in was_printed:
                                continue
                            was_printed.add(i)
                            out_gene, translation = translations[i]
                            out.write(f">{out_gene}\n{translation}\n")


def change_fasta_start_pos(input_fasta: str, shift: int, output_fasta: str = ""):
    """
    Shifts the start position of sequences in a FASTA file by a given number of positions.

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file.

    shift : int
        Number of positions to shift the start position.

    output_fasta : str, optional
        Path to the output shifted FASTA file. If not provided, the output file will be named
        based on the input file with a suffix '_shifted.fasta'.
    """
    if output_fasta == "":
        output_fasta = os.path.basename(input_fasta)
        output_fasta = output_fasta.replace(".fasta", "_shifted.fasta")
    output_path = os.path.join("./", output_fasta)
    with open(input_fasta, "r") as input:
        with open(output_path, "w"):
            while True:
                line = input.readline().strip()
                print(line)
                if not line:
                    break
                if not line.startswith(">"):
                    line = line[shift:] + line[:shift]
                with open(output_path, "a") as output:
                    output.write(line + "\n")
