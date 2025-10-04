import logging
import re
from typing import Any
from ..constants.constants import *

logger = logging.getLogger(__name__)


class GeneExtractor:
    def extract_gene_names_from_results(self, results: dict[str, Any]) -> list[str]:
        try:
            db_results = (
                results.get("results", {}).get("combined_analysis", {}).get("database_results")
                or results.get("combined_analysis", {}).get("database_results")
                or results.get("database_results")
            )

            gene_names: list[str] = []
            if isinstance(db_results, dict):
                gene_names.extend(self._extract_from_database_results(db_results))
            elif isinstance(results.get("results"), str):
                gene_names.extend(self._extract_from_text(results["results"]))

            return self._clean_gene_names(gene_names)
        except Exception as e:
            logger.warning(f"Error extracting gene names: {e}")
            return []
    
    def _extract_from_database_results(self, db_results: dict[str, Any]) -> list[str]:
        return self._extract_best_gene_names(db_results)
    
    def _extract_best_gene_names(self, db_results: dict[str, Any]) -> list[str]:
        gene_names = []
        logger.info(f"Extracting from database results: {list(db_results.keys())}")
        
        for result_type in ['ncbi_protein_results', 'ncbi_nucleotide_results', 'uniprot_results']:
            if result_type in db_results:
                logger.info(f"Found {result_type} with {len(db_results[result_type])} hits")
                for hit in db_results[result_type]:
                    if isinstance(hit, dict):
                        best_name = self._get_best_gene_name_from_hit(hit)
                        logger.info(f"  Hit gene name: {best_name} from {hit.get('description', NO_DESCRIPTION)[:LOG_DESC_PREVIEW_SHORT]}...")
                        if best_name and best_name not in gene_names:
                            gene_names.append(best_name)
        
        logger.info(f"Extracted {len(gene_names)} unique gene names: {gene_names}")
        return gene_names
    
    def _get_best_gene_name_from_hit(self, hit: dict[str, Any]) -> str:
        gene_name = str(hit.get('gene_name', ''))
        if gene_name and gene_name != 'Unknown' and not self._is_protein_id(gene_name):
            return gene_name
        
        description = hit.get('description', '')
        if description:
            bracket_matches: list[str] = re.findall(r'\[gene=([^\]]+)\]', description)
            for match in bracket_matches:
                if match and not self._is_protein_id(match):
                    return match.strip()
            
            full_matches: list[str] = re.findall(r'(?:Full|GN)=([^;\]]+)', description)
            for match in full_matches:
                if match and not self._is_protein_id(match):
                    clean_name = match.strip()
                    if ' ' in clean_name:
                        parts = clean_name.split()
                        for part in reversed(parts):
                            if (len(part) >= GENE_EXTRACTOR_MIN_GENE_LENGTH and 
                                not self._is_protein_id(part) and 
                                part.upper() not in ['PROTEIN', 'PUTATIVE', 'UNCHARACTERIZED', 'COMMUNIS', 'FILAMENT-BINDING', 'MURC', 'DDL']):
                                return part
                    return clean_name
            
            gene_patterns: list[str] = re.findall(r'\b([A-Z]{' + str(GENE_EXTRACTOR_MIN_PATTERN_LENGTH) + ',}[A-Z0-9]*)\b', description)
            for match in gene_patterns:
                if not self._is_protein_id(match):
                    return match
            
            gene_specific_matches: list[str] = re.findall(r'(?:gene|protein)\s+([A-Z][A-Z0-9]{' + str(GENE_EXTRACTOR_MIN_SPECIFIC_LENGTH) + ',})', description, re.IGNORECASE)
            for match in gene_specific_matches:
                if not self._is_protein_id(match):
                    return match.upper()
            
            gene_specific_matches2: list[str] = re.findall(r'([A-Z][A-Z0-9]{' + str(GENE_EXTRACTOR_MIN_SPECIFIC_LENGTH) + ',})\s+(?:gene|protein)', description, re.IGNORECASE)
            for match in gene_specific_matches2:
                if not self._is_protein_id(match):
                    return match.upper()
            
            short_matches: list[str] = re.findall(r'Short=([A-Z][A-Z0-9]{' + str(GENE_EXTRACTOR_MIN_SPECIFIC_LENGTH) + ',})', description)
            for match in short_matches:
                if not self._is_protein_id(match):
                    return match.upper()
            
            altname_matches: list[str] = re.findall(r'AltName:\s*Full=([A-Z][A-Z0-9]{' + str(GENE_EXTRACTOR_MIN_SPECIFIC_LENGTH) + ',})', description)
            for match in altname_matches:
                if not self._is_protein_id(match):
                    return match.upper()
        
        return ""
    
    def _is_protein_id(self, name: str) -> bool:
        if not name or len(name) < GENE_EXTRACTOR_MIN_GENE_LENGTH:
            return True
        
        if re.match(r'^[a-z]\d[a-z0-9]{' + str(GENE_EXTRACTOR_PROTEIN_ID_MIN_LENGTH) + ',' + str(GENE_EXTRACTOR_PROTEIN_ID_MAX_LENGTH) + '}(\.\d+)?$', name.lower()):
            return True
        
        if re.match(r'^pro\d+$', name.lower()) and len(name) <= GENE_EXTRACTOR_MAX_PRO_LENGTH:
            return True
        
        if sum(c.isdigit() for c in name) > len(name) * GENE_EXTRACTOR_DIGIT_THRESHOLD:
            return True
        
        if name.lower() in ['unknown', 'protein', 'putative', 'uncharacterized', 'pro', 'communis', 'filament-binding', 'murc', 'ddl', 'gvqw1', 'subunit', 'cadherin-20', 'c16orf89']:
            return True
        
        if name.lower() in ['binding', 'transporter', 'synthase', 'ligase', 'transferase', 'reductase', 'oxidase', 'dehydrogenase', 'subunit', 'complex', 'assembly']:
            return True
        
        return bool(re.match(r'^[A-Z]\d[A-Z0-9]{' + str(GENE_EXTRACTOR_LONG_ID_MIN_LENGTH) + ',}(\.\d+)?$', name))
    
    def _extract_from_text(self, text: str) -> list[str]:
        gene_names = []
        
        uniprot_matches = re.findall(r'([A-Z]\d[A-Z0-9]{' + str(GENE_EXTRACTOR_PROTEIN_ID_MIN_LENGTH) + ',' + str(GENE_EXTRACTOR_PROTEIN_ID_MAX_LENGTH) + '}\.\d+)', text)
        for match in uniprot_matches:
            base_id = match.split('.')[GENE_EXTRACTOR_DOT_SPLIT_INDEX] if '.' in match else match
            if base_id not in gene_names:
                gene_names.append(base_id)
        
        return gene_names
    
    def _clean_gene_names(self, gene_names: list[str]) -> list[str]:
        cleaned_names = []
        for name in gene_names:
            if not name or len(name) < GENE_EXTRACTOR_CLEAN_MIN_LENGTH:
                continue
                
            clean_name = re.sub(r'\s*\[[^\]]*\]\s*$', '', name)
            
            clean_name = re.sub(r'^(gene_|protein_|hypothetical_|predicted_|putative_)', '', clean_name.lower())
            clean_name = re.sub(r'(_gene|_protein|_predicted|_putative)$', '', clean_name)
            clean_name = clean_name.upper()
            
            clean_name = clean_name.strip()
            
            if (clean_name in ['HOMO', 'SAPIENS', 'PROTEIN', 'PUTATIVE', 'UNCHARACTERIZED', 'PRO'] or
                re.match(r'^[A-Z]\d[A-Z0-9]{' + str(GENE_EXTRACTOR_PROTEIN_ID_MIN_LENGTH) + ',' + str(GENE_EXTRACTOR_PROTEIN_ID_MAX_LENGTH) + '}(\.\d+)?$', clean_name) or
                (re.match(r'^PRO\d+$', clean_name) and len(clean_name) <= GENE_EXTRACTOR_MAX_PRO_LENGTH) or
                len(clean_name) < GENE_EXTRACTOR_MIN_GENE_LENGTH or
                sum(c.isdigit() for c in clean_name) > len(clean_name) * GENE_EXTRACTOR_DIGIT_THRESHOLD):
                continue
                
            if len(clean_name) >= GENE_EXTRACTOR_MIN_GENE_LENGTH and clean_name not in cleaned_names:
                cleaned_names.append(clean_name)
        
        return cleaned_names
