from abc import ABC, abstractmethod
from typing import Dict
from bio_files_processor import (
    extract_exons,
    extract_introns,
    extract_predicted_peptides,
    GenscanOutput,
)
from bs4 import BeautifulSoup
import requests

# HW17


def run_genscan(
    sequence=None,
    sequence_file=None,
    organism="Vertebrate",
    exon_cutoff=1.00,
    sequence_name="",
):
    """
    Runs the GENSCAN prediction on the given sequence or sequence file.

    Parameters
    ----------
    sequence : str, optional
        The sequence to be analyzed.

    sequence_file : str, optional
        The file path of the sequence to be analyzed.

    organism : str, default="Vertebrate"
        The organism type (Vertebrate, Arabidopsis, Maize).

    exon_cutoff : float, default=1.00
        The exon cutoff value for GENSCAN predictions.

    sequence_name : str, default=""
        The name of the sequence.

    Returns
    -------
    GenscanOutput
        The output of the GENSCAN prediction containing status code, CDS list, intron list, and exon list.

    Raises
    ------
    ValueError
        If required parameters are not provided or if invalid values are given.
    """
    # Check user-input
    if not (sequence_file or sequence):
        raise ValueError("Please provide either sequence or sequence_file")

    if not organism:
        raise ValueError("Please provide organism")

    if sequence_file and sequence:
        raise ValueError(
            "You provided both sequence_file and sequence. Please provide one of two"
        )

    if organism not in ["Vertebrate", "Arabidopsis", "Maize"]:
        raise ValueError(
            "Please provide one of the supported organism subphylums: Arabidopsis, Vertebrate, Maize"
        )
    # Read sequence from file if provided
    if sequence_file:
        with open(sequence_file, "r") as file:
            sequence = file.read().strip()

    # Prepare parameters for the POST request
    params = {
        "-o": organism,
        "-s": sequence,
        "-n": sequence_name,
        "-e": exon_cutoff,
        "-p": "Predicted peptides only",
    }

    # Send POST request to GENSCAN service
    url = "http://argonaute.mit.edu/cgi-bin/genscanw_py.cgi"
    response = requests.post(url, data=params)
    status = response.status_code

    # Check if connection was successful
    if status != 200:
        raise ValueError("Failed to connect to GENSCAN service")

    # Parse the HTML response
    soup = BeautifulSoup(response.text, "html.parser")
    pre_tag = soup.find("pre")
    pre_text = pre_tag.get_text()
    lines = pre_text.split("\n")
    exon_list = []
    intron_list = []
    cds_list = []

    # Extract predicted exons
    if "NO EXONS/GENES PREDICTED IN SEQUENCE" not in lines:
        exon_list = extract_exons(lines)
    # Extract predicted introns
    if "NO EXONS/GENES PREDICTED IN SEQUENCE" not in lines:
        intron_list = extract_introns(lines)
    # Extract predicted peptides
    if "NO PEPTIDES PREDICTED" not in lines:
        cds_list = extract_predicted_peptides(lines)
    return GenscanOutput(status, cds_list, intron_list, exon_list)


# HW14


class BiologicalSequence(ABC):
    """
    Abstract base class representing a biological sequence.

    Attributes
    ----------
    sequence : str
        The biological sequence.
    """

    def __init__(self, sequence: str):
        self.sequence = sequence

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, slc):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def is_valid_alphabet(self) -> bool:
        pass


class NucleicAcid(BiologicalSequence):
    """
    Represents a nucleic acid sequence.

    Attributes
    ----------
    sequence : str
        The nucleic acid sequence.
    """

    ALPHABET: set = set()
    MAP: Dict[str, str] = {}

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, slc):
        return self.sequence[slc]

    def __str__(self):
        return self.sequence

    def __repr__(self):
        return f"NucleicAcid(sequence={self.sequence})"

    def is_valid_alphabet(self) -> bool:
        """
        Check if the sequence contains a valid nucleotide alphabet.

        Returns
        -------
        bool
            True if the sequence contains valid nucleotides, False otherwise.
        """
        return set(self.sequence).issubset(self.ALPHABET)

    def complement(self):
        """
        Generate the complementary sequence.

        Returns
        -------
        NucleicAcid
            The complementary sequence.

        Raises
        ------
        NotImplementedError
            If called on a base NucleicAcid instance.
        """
        if type(self) is NucleicAcid:
            raise NotImplementedError(
                "Cannot complement a generic NucleicAcid instance"
            )

        comp_seq = "".join(self.MAP[base] for base in self.sequence)
        return type(self)(comp_seq)

    def gc_content(self) -> float:
        """
        Calculate the GC content of the sequence.

        Returns
        -------
        float
            The GC content percentage.

        Raises
        ------
        NotImplementedError
            If called on a base NucleicAcid instance.
        """
        if type(self) is NucleicAcid:
            raise NotImplementedError(
                "Cannot calculate GC content for a generic NucleicAcid instance"
            )

        gc_count = sum(base in "GC" for base in self.sequence)
        return (gc_count / len(self.sequence)) * 100


class DNASequence(NucleicAcid):
    """
    Represents a DNA sequence.

    Attributes
    ----------
    sequence : str
        The DNA sequence.
    """

    ALPHABET = set("ATGC")
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}

    def transcribe(self):
        """
        Transcribe the DNA sequence into RNA.

        Returns
        -------
        RNASequence
            The transcribed RNA sequence.
        """
        transcribed_sequence = self.sequence.replace("T", "U")
        return RNASequence(transcribed_sequence)


class RNASequence(NucleicAcid):
    """
    Represents an RNA sequence.

    Attributes
    ----------
    sequence : str
        The RNA sequence.
    """

    ALPHABET = set("AUGC")
    MAP = {"A": "U", "U": "A", "C": "G", "G": "C"}


class AminoAcidSequence(BiologicalSequence):
    """
    Represents an amino acid sequence.

    Attributes
    ----------
    sequence : str
        The amino acid sequence.
    """

    ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, slc):
        return self.sequence[slc]

    def __str__(self):
        return self.sequence

    def __repr__(self):
        return f"AminoAcidSequence(sequence={self.sequence})"

    def is_valid_alphabet(self) -> bool:
        """
        Check if the sequence contains a valid amino acid alphabet.

        Returns
        -------
        bool
            True if the sequence contains valid amino acids, False otherwise.
        """
        return set(self.sequence).issubset(self.ALPHABET)

    def calculate_aa_freq(self) -> Dict[str, int]:
        """
        Calculate the frequency of each amino acid in the sequence.

        Returns
        -------
        dict
            Dictionary with the frequency of each amino acid.
        """
        aa_freq = {}
        for amino_acid in self.sequence:
            if amino_acid in aa_freq:
                aa_freq[amino_acid] += 1
            else:
                aa_freq[amino_acid] = 1
        return aa_freq
