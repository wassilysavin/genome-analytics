import logging
import re
from typing import Any, Optional

from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from ...models.bio_models import ORF, ProteinProperties, SequenceComposition
from ...constants.constants import *

logger = logging.getLogger(__name__)


def find_orfs(sequence: str, min_length: int = BIO_MIN_ORF_LENGTH) -> list[dict[str, Any]]:
    try:
        clean_sequence = _clean_and_validate_sequence(sequence)
        if not clean_sequence:
            return []

        seq_obj = Seq(clean_sequence)
        orfs = []

        # Check all 6 reading frames (3 forward, 3 reverse)
        for strand, nuc in [(1, seq_obj), (-1, seq_obj.reverse_complement())]:
            for frame in range(3):
                frame_orfs = _find_orfs_in_frame(
                    nuc, frame, strand, min_length, len(clean_sequence)
                )
                orfs.extend(frame_orfs)

        orfs.sort(key=lambda x: x.length, reverse=True)
        orf_dicts = [_orf_to_dict(orf) for orf in orfs]

        # logger.info(f"Found {len(orf_dicts)} ORFs >= {min_length} bp")
        return orf_dicts

    except Exception as e:
        logger.error(f"ORF detection failed: {e}")
        return []


def _clean_and_validate_sequence(sequence: str) -> Optional[str]:
    if not sequence:
        logger.error("Empty sequence provided")
        return None

    clean_sequence = sequence.upper().replace(" ", "").replace("\n", "").replace("\t", "")

    # Validate sequence contains only DNA nucleotides
    if not re.match(r"^[ATCG]+$", clean_sequence):
        logger.error("Invalid DNA sequence: contains non-DNA characters")
        return None

    return clean_sequence


def _find_orfs_in_frame(
    nuc_seq: Seq, frame: int, strand: int, min_length: int, seq_length: int
) -> list[ORF]:
    frame_seq = str(nuc_seq[frame:])
    orfs = []

    i = 0
    while i < len(frame_seq) - min_length:
        # Look for start codon
        if _is_start_codon(frame_seq[i : i + 3]):
            # Look for stop codon
            for j in range(i + 3, len(frame_seq), 3):
                if _is_stop_codon(frame_seq[j : j + 3]):
                    orf_length = j - i
                    if orf_length >= min_length:
                        orf = _create_orf(frame_seq, i, j, frame, strand, seq_length)
                        orfs.append(orf)
                    break
        i += 3

    return orfs


def _is_start_codon(codon: str) -> bool:
    return codon in BIO_START_CODONS


def _is_stop_codon(codon: str) -> bool:
    return codon in BIO_STOP_CODONS


def _create_orf(
    frame_seq: str, start_idx: int, end_idx: int, frame: int, strand: int, seq_length: int
) -> ORF:
    orf_length = end_idx - start_idx

    # Calculate actual positions
    if strand == 1:
        start_pos = start_idx + frame + 1
        end_pos = end_idx + frame + 3
    else:
        start_pos = seq_length - (end_idx + frame + 3) + 1
        end_pos = seq_length - (start_idx + frame) + 1

    return ORF(
        start=start_pos,
        end=end_pos,
        length=orf_length,
        strand="+" if strand == 1 else "-",
        frame=frame + 1,
        sequence=frame_seq[start_idx : end_idx + 3],
        start_codon=frame_seq[start_idx : start_idx + 3],
    )


def _orf_to_dict(orf: ORF) -> dict[str, Any]:
    return {
        "start": orf.start,
        "end": orf.end,
        "length": orf.length,
        "strand": orf.strand,
        "frame": orf.frame,
        "sequence": orf.sequence,
        "start_codon": orf.start_codon,
    }


def analyze_protein_properties(protein_sequence: str) -> dict[str, Any]:
    try:
        clean_seq = _clean_protein_sequence(protein_sequence)
        if not clean_seq:
            return {"error": "Invalid protein sequence"}

        properties = _calculate_protein_properties(clean_seq)
        return _protein_properties_to_dict(properties)

    except Exception as e:
        logger.error(f"Protein analysis failed: {e}")
        return {"error": f"Analysis failed: {str(e)}"}


def _clean_protein_sequence(protein_sequence: str) -> Optional[str]:
    if not protein_sequence or len(protein_sequence) < BIO_MIN_PROTEIN_LENGTH:
        logger.error("Protein sequence too short for analysis")
        return None

    # Remove stop codons (*)
    clean_seq = protein_sequence.replace("*", "")

    if not clean_seq:
        logger.error("No valid amino acids found")
        return None

    return clean_seq


def _calculate_protein_properties(clean_seq: str) -> ProteinProperties:
    analysis = ProteinAnalysis(clean_seq)

    molecular_weight = round(analysis.molecular_weight(), 2)
    isoelectric_point = round(analysis.isoelectric_point(), 2)
    instability_index = round(analysis.instability_index(), 2)
    aromaticity = round(analysis.aromaticity(), 3)
    gravy = round(analysis.gravy(), 3)

    amino_acid_composition = {
        aa: round(percent, 2) for aa, percent in analysis.get_amino_acids_percent().items()
    }

    secondary_structure = analysis.secondary_structure_fraction()

    stability = "Stable" if instability_index < BIO_INSTABILITY_STABLE_THRESHOLD else "Unstable"
    hydrophobicity = "Hydrophobic" if gravy > BIO_HYDROPHOBIC_THRESHOLD else "Hydrophilic"

    return ProteinProperties(
        length=len(clean_seq),
        molecular_weight=molecular_weight,
        isoelectric_point=isoelectric_point,
        instability_index=instability_index,
        aromaticity=aromaticity,
        amino_acid_composition=amino_acid_composition,
        gravy=gravy,
        secondary_structure=secondary_structure,
        stability=stability,
        hydrophobicity=hydrophobicity,
    )


def _protein_properties_to_dict(properties: ProteinProperties) -> dict[str, Any]:
    return {
        "length": properties.length,
        "molecular_weight": properties.molecular_weight,
        "isoelectric_point": properties.isoelectric_point,
        "instability_index": properties.instability_index,
        "aromaticity": properties.aromaticity,
        "amino_acid_composition": properties.amino_acid_composition,
        "gravy": properties.gravy,
        "secondary_structure": properties.secondary_structure,
        "stability": properties.stability,
        "hydrophobicity": properties.hydrophobicity,
    }


def calculate_sequence_composition(sequence: str) -> dict[str, Any]:
    try:
        clean_seq = _clean_sequence_for_composition(sequence)
        if not clean_seq:
            return {"error": "No valid nucleotides found"}

        composition = _calculate_composition_stats(clean_seq)
        return _composition_to_dict(composition)

    except Exception as e:
        logger.error(f"Sequence composition analysis failed: {e}")
        return {"error": f"Analysis failed: {str(e)}"}


def _clean_sequence_for_composition(sequence: str) -> Optional[str]:
    if not sequence:
        return None

    clean_seq = sequence.upper().replace("N", "")

    if not clean_seq:
        return None

    return clean_seq


def _calculate_composition_stats(clean_seq: str) -> SequenceComposition:
    gc_content = round(gc_fraction(clean_seq) * 100, 2)

    nucleotide_counts = {
        "A": clean_seq.count("A"),
        "T": clean_seq.count("T"),
        "G": clean_seq.count("G"),
        "C": clean_seq.count("C"),
    }

    total = len(clean_seq)
    nucleotide_percentages = {
        base: round((count / total) * 100, 2) for base, count in nucleotide_counts.items()
    }

    if gc_content < BIO_GC_LOW_THRESHOLD:
        gc_assessment = "Low GC content"
    elif gc_content > BIO_GC_HIGH_THRESHOLD:
        gc_assessment = "High GC content"
    else:
        gc_assessment = "Optimal GC content"

    return SequenceComposition(
        length=len(clean_seq),
        gc_content=gc_content,
        nucleotide_counts=nucleotide_counts,
        nucleotide_percentages=nucleotide_percentages,
        gc_assessment=gc_assessment,
    )


def _composition_to_dict(composition: SequenceComposition) -> dict[str, Any]:
    return {
        "length": composition.length,
        "gc_content": composition.gc_content,
        "nucleotide_counts": composition.nucleotide_counts,
        "nucleotide_percentages": composition.nucleotide_percentages,
        "gc_assessment": composition.gc_assessment,
    }


def translate_orf(orf_sequence: str) -> str:
    try:
        clean_seq = _clean_and_validate_sequence(orf_sequence)
        if not clean_seq:
            logger.error("Invalid ORF sequence for translation")
            return ""

        # Translate to protein
        seq_obj = Seq(clean_seq)
        protein = str(seq_obj.translate())

        logger.debug(f"Translated ORF ({len(clean_seq)} bp) to protein ({len(protein)} aa)")
        return protein

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return ""
