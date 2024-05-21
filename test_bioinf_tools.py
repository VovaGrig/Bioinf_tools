from bio_files_processor import (
    OpenFasta,
    FastaRecord,
    GenscanOutput,
    convert_multiline_fasta_to_oneline,
    change_fasta_start_pos,
)
from bioinf_tools import AminoAcidSequence, run_genscan
import pytest
import os


@pytest.fixture
def tmp_output_file():
    file_path = "tmp_output.fasta"
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture
def tmp_fasta_for_testing_change_fasta_start_pos():
    file_path = "tmp_for_testing_change_fasta_start_pos.fasta"
    with open(file_path, "w") as f:
        f.write(">Genome\nCGTAT\n>Genome2\nADW@D\n")
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)


def test_change_fasta_start_pos_exists(
    tmp_fasta_for_testing_change_fasta_start_pos, tmp_output_file
):
    """
    Test change_fasta_start_pos creates output_fasta
    """
    change_fasta_start_pos(
        input_fasta=tmp_fasta_for_testing_change_fasta_start_pos,
        shift=1,
        output_fasta=tmp_output_file,
    )
    assert os.path.exists(tmp_output_file)


def test_change_fasta_start_pos_content(
    tmp_fasta_for_testing_change_fasta_start_pos, tmp_output_file
):
    """
    Test change_fasta_start_pos output_fasta content
    """
    change_fasta_start_pos(
        input_fasta=tmp_fasta_for_testing_change_fasta_start_pos,
        shift=1,
        output_fasta=tmp_output_file,
    )
    with open(tmp_output_file, "r") as output_fasta:
        print(output_fasta.readlines)
        assert output_fasta.readlines() == [
            ">Genome\n",
            "GTATC\n",
            ">Genome2\n",
            "DW@DA\n",
        ]


@pytest.fixture
def tmp_fasta_for_testing_convert_multiline_fasta_to_oneline():
    file_path = "tmp_for_testing_convert_multiline_fasta_to_oneline.fasta"
    with open(file_path, "w") as f:
        f.write(">Genome\nCGTAT\nCGTAT\n>Genome2\nADW@D\nADW@D\n")
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)


def test_convert_multiline_fasta_to_oneline_content(
    tmp_fasta_for_testing_convert_multiline_fasta_to_oneline, tmp_output_file
):
    convert_multiline_fasta_to_oneline(
        input_fasta=tmp_fasta_for_testing_convert_multiline_fasta_to_oneline,
        output_fasta=tmp_output_file,
    )
    with open(tmp_output_file, "r") as output_fasta:
        assert output_fasta.readlines() == [
            ">Genome\n",
            "CGTATCGTAT\n",
            ">Genome2\n",
            "ADW@DADW@D",
        ]


def test_calculate_aa_freq():
    sequence = "CASSQDTEVFF"
    aa_seq = AminoAcidSequence(sequence)
    aa_freq = aa_seq.calculate_aa_freq()
    assert aa_freq["F"] == 2 & aa_freq["S"] == 2


def test_run_genscan_double_sequence_provided_error_raise():
    with pytest.raises(ValueError):
        run_genscan(sequence="ATCG", sequence_file="data/example_gene_for_genscan.fna")


def test_run_genscan_output_instance():
    assert isinstance(
        run_genscan(sequence_file="data/example_gene_for_genscan.fna"), GenscanOutput
    )


@pytest.fixture
def input_data_test_OpenFasta() -> list:
    """
    Fixture for input fasta data.
    """
    fasta = [
        FastaRecord(
            id=">GTD323452",
            description="5S_rRNA NODE_272_length_223_cov_0.720238:18-129(+)",
            seq="ACGGCCATAGGACTTTGAAAGCACCGCATCCCGTCCGATCTGCGAAGTTAACCAAGATGCCGCCTGGTTAGTACCATGGTGGGGGACCACATGGGAATCCCTGGTGCTGTG",
        ),
        FastaRecord(
            id=">GTD678345",
            description="16S_rRNA NODE_80_length_720_cov_1.094737:313-719(+)",
            seq="TTGGCTTCTTAGAGGGACTTTTGATGTTTAATCAAAGGAAGTTTGAGGCAATAACAGGTCTGTGATGCCCTTAGATGTTCTGGGCCGCACGCGCGCTACACTGAGCCCTTGGGAGTGGTCCATTTGAGCCGGCAACGGCACGTTTGGACTGCAAACTTGGGCAAACTTGGTCATTTAGAGGAAGTAAAAGTCGTAACAAGGT",
        ),
        FastaRecord(
            id=">GTD174893",
            description="16S_rRNA NODE_1_length_2558431_cov_75.185164:2153860-2155398(+)",
            seq="TTGAAGAGTTTGATCATGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGTAACAGGAAACAGCTTGCTGTTTCGCTGACGAGTGGGAAGTAGGTAGCTTAACCTTCGGGAGGGCGCTTACCACTTTGTGATTCATGACTGGGGTGAAGTCGTAACAAGGTAACCGTAGGGGAACCTGCGGTTGGATCACCTCCTT",
        ),
        FastaRecord(
            id=">GTD906783",
            description="16S_rRNA NODE_1_length_2558431_cov_75.185164:793941-795479(-)",
            seq="TTGAAGAGTTTGATCATGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGTAACAGGAAACAGCTTGCTGTTTCGCTGACGAGTGGGAAGTAGGTAGCTTAACCTTCGGGAGGGCGCTTACCACTTTGTGATTCATGACTGGGGTGAAGTCGTAACAAGGTAACCGTAGGGGAACCTGCGGTTGGATCACCTCCTT",
        ),
        FastaRecord(
            id=">GTD129563",
            description="16S_rRNA NODE_4_length_428221_cov_75.638017:281055-282593(-)",
            seq="CGGACGGGTGAGTAATGTCTGGGAAACTGCCTGATGGAGGGGGATAACTACTGGAAACGGTAGCTAATACCGCATAACGTCGCAAGACCAAAGAGGGGGACCGAAGTAGGTAGCTTAACCTTCGGGAGGGCGCTTACCACTTTGTGATTCATGACTGGGGTGAAGTCGTAACAAGGTAACCGTAGGGGAACCTGCGGTTGGATCACCTCCTT",
        ),
    ]
    return fasta


def test_OpenFasta_output_len(input_data_test_OpenFasta):
    with OpenFasta("data/example_fasta.fasta") as f:
        total_fasta = f.read_records()
    assert len(total_fasta) == len(input_data_test_OpenFasta)


def test_OpenFasta_output_content(input_data_test_OpenFasta):
    with OpenFasta("data/example_fasta.fasta") as f:
        total_fasta = f.read_records()
    assert total_fasta == input_data_test_OpenFasta
