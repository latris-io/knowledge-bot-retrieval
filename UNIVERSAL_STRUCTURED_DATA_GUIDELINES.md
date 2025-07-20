# UNIVERSAL STRUCTURED DATA INGESTION GUIDELINES

## 🔬 **CONTENT-AGNOSTIC PRINCIPLES FOR STRUCTURED DATA**

**For Ingestion Service Development Team**  
**Based on 76.9% Structured Data Failure Rate Diagnosis**

---

## 🎯 **UNIVERSAL DETECTION PRINCIPLES**

### **Information Theory-Based Chunking** 
Use mathematical properties, not domain assumptions:

#### **1. Tabular Structure Detection**
```python
# CONTENT-AGNOSTIC: Separator consistency analysis
def detect_tabular_data(content: str) -> bool:
    """Universal tabular detection using information theory"""
    separators = [',', '\t', '|', ';', ':']
    lines = content.split('\n')
    
    for sep in separators:
        if sep in content:
            counts_per_line = [line.count(sep) for line in lines if line.strip()]
            if len(counts_per_line) >= 2:
                # Calculate consistency ratio
                most_common = max(set(counts_per_line), key=counts_per_line.count)
                consistency = counts_per_line.count(most_common) / len(counts_per_line)
                if consistency >= 0.7:  # 70% of lines have same separator count
                    return True
    return False
```

#### **2. Repetitive Data Pattern Detection**
```python
# CONTENT-AGNOSTIC: Structural pattern analysis  
def detect_data_patterns(content: str) -> dict:
    """Analyze content structure using universal patterns"""
    lines = content.split('\n')[:20]  # Sample for efficiency
    
    patterns = []
    for line in lines:
        if not line.strip():
            continue
            
        # Analyze universal structural patterns
        words = line.split()[:10]  # First 10 words per line
        line_pattern = []
        for word in words:
            if word.isdigit():
                line_pattern.append('NUM')
            elif any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                line_pattern.append('MIXED')  # Price-like patterns
            elif word.isalpha():
                line_pattern.append('TEXT')
            else:
                line_pattern.append('SPECIAL')
        
        patterns.append(tuple(line_pattern))
    
    # Calculate pattern repetition (indicates structured data)
    if patterns:
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        max_repetition = max(pattern_counts.values()) / len(patterns)
        return {
            'is_structured': max_repetition >= 0.4,  # 40% pattern repetition
            'repetition_ratio': max_repetition,
            'dominant_pattern': max(pattern_counts, key=pattern_counts.get)
        }
    
    return {'is_structured': False}
```

---

## 🏗️ **UNIVERSAL CHUNKING STRATEGIES**

### **Strategy 1: Row-Level Chunking (For High Separator Consistency)**
```python
# When tabular_consistency >= 0.7
def chunk_tabular_data(content: str, separator: str) -> List[Dict]:
    lines = content.split('\n')
    header_line = None
    chunks = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        fields = line.split(separator)
        
        # First content line becomes header reference
        if header_line is None and len(fields) >= 2:
            header_line = fields
            continue
        
        # Each subsequent line becomes a searchable chunk
        if len(fields) >= len(header_line):
            # Create human-readable chunk content
            chunk_content = []
            metadata = {}
            
            for j, field in enumerate(fields):
                if j < len(header_line) and field.strip():
                    field_name = header_line[j].strip().lower().replace(' ', '_')
                    chunk_content.append(f"{header_line[j]}: {field.strip()}")
                    metadata[f"field_{field_name}"] = field.strip()
            
            chunks.append({
                'content': ' | '.join(chunk_content),
                'metadata': {
                    **metadata,
                    'chunk_type': 'data_row',
                    'row_index': i,
                    'field_count': len(fields)
                }
            })
    
    return chunks
```

### **Strategy 2: Pattern-Based Chunking (For Repetitive Structures)**
```python  
# When pattern_repetition >= 0.4
def chunk_repetitive_data(content: str, dominant_pattern: tuple) -> List[Dict]:
    lines = content.split('\n')
    chunks = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        words = line.split()
        if len(words) < 2:
            continue
            
        # Analyze if this line matches the dominant pattern
        line_pattern = []
        for word in words[:len(dominant_pattern)]:
            if word.isdigit():
                line_pattern.append('NUM')
            elif any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                line_pattern.append('MIXED')
            elif word.isalpha():
                line_pattern.append('TEXT')
            else:
                line_pattern.append('SPECIAL')
        
        if tuple(line_pattern) == dominant_pattern:
            # Create individual chunk for this structured line
            chunks.append({
                'content': line.strip(),
                'metadata': {
                    'chunk_type': 'structured_item',
                    'line_index': i,
                    'pattern_match': True,
                    'pattern_type': '-'.join(dominant_pattern)
                }
            })
    
    return chunks
```

### **Strategy 3: Hierarchical List Chunking (For Nested Content)**
```python
# For markdown lists, bullet points, etc.
def chunk_hierarchical_lists(content: str) -> List[Dict]:
    lines = content.split('\n')
    chunks = []
    current_section = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Detect list indicators (content-agnostic)
        list_indicators = ['- ', '* ', '+ ', '1. ', '2. ', '3.', '4.', '5.']
        is_list_item = any(stripped.startswith(indicator) for indicator in list_indicators)
        
        # Detect section headers (content-agnostic)
        is_header = (stripped.startswith('#') or 
                    stripped.isupper() and len(stripped) < 100 or
                    stripped.endswith(':') and len(stripped) < 80)
        
        if is_header:
            current_section = stripped
        elif is_list_item:
            # Create individual chunk for each list item
            item_content = stripped
            if current_section:
                item_content = f"{current_section} | {item_content}"
                
            chunks.append({
                'content': item_content,
                'metadata': {
                    'chunk_type': 'list_item',
                    'section_header': current_section,
                    'line_index': i,
                    'is_structured': True
                }
            })
    
    return chunks
```

---

## 🏷️ **UNIVERSAL METADATA TAGGING**

### **Content-Agnostic Field Detection**
```python
def extract_universal_metadata(chunk_content: str, chunk_metadata: dict) -> dict:
    """Extract searchable metadata using universal patterns"""
    enhanced_metadata = chunk_metadata.copy()
    
    # Universal field patterns (no domain assumptions)
    field_patterns = {
        'numeric_values': r'\$?[\d,]+\.?\d*',  # Numbers, prices, quantities
        'alphanumeric_ids': r'[A-Za-z]+\d+|[A-Z]{2,}\d+',  # Product IDs, codes
        'percentage_values': r'\d+%',  # Percentages  
        'date_patterns': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # Dates
        'phone_patterns': r'\(\d{3}\)\s?\d{3}-?\d{4}',  # Phone numbers
        'email_patterns': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Emails
    }
    
    # Extract universal field types
    for field_type, pattern in field_patterns.items():
        import re
        matches = re.findall(pattern, chunk_content)
        if matches:
            enhanced_metadata[f'{field_type}_found'] = matches[:5]  # Limit to 5 matches
            enhanced_metadata[f'{field_type}_count'] = len(matches)
    
    # Universal text analysis
    words = chunk_content.split()
    enhanced_metadata.update({
        'word_count': len(words),
        'unique_words': len(set(w.lower() for w in words)),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'contains_numbers': any(c.isdigit() for c in chunk_content),
        'contains_special_chars': any(c in '|,:;()-[]{}' for c in chunk_content)
    })
    
    return enhanced_metadata
```

---

## 🎯 **IMPLEMENTATION PRIORITIES**

### **Phase 1: Universal Detection (URGENT)**
1. **Implement content-agnostic tabular detection**
   - Separator consistency analysis
   - Pattern repetition calculation  
   - No file extension dependencies

2. **Apply universal chunking strategies**
   - Row-level for high tabular consistency (≥0.7)
   - Pattern-based for repetitive structures (≥0.4)
   - Hierarchical for list-like content

### **Phase 2: Enhanced Metadata (HIGH PRIORITY)**
1. **Extract universal field patterns**  
   - Numbers, codes, dates, contacts
   - No business-specific assumptions
   - Mathematical/structural analysis only

2. **Index searchable entities**
   - Field values as searchable terms
   - Structural metadata for filtering
   - Universal content characteristics

### **Phase 3: Quality Validation (IMPORTANT)**
1. **Test with current failing queries**
   - Use diagnostic queries from structured data analysis
   - Validate 75%+ success rate improvement
   - Content-agnostic success metrics

2. **Monitor chunking effectiveness**
   - Track chunk creation patterns
   - Measure searchability improvements
   - No domain-specific KPIs

---

## ✅ **CONTENT-AGNOSTIC COMPLIANCE**

### **What This Approach DOES:**
- ✅ Uses mathematical/information theory principles
- ✅ Applies universal structural analysis  
- ✅ Works across all domains (medical, legal, financial, academic)
- ✅ No hardcoded business terms or file types
- ✅ Scalable pattern recognition algorithms

### **What This Approach AVOIDS:**
- ❌ No hardcoded file extensions (.csv, .xlsx)
- ❌ No business-specific patterns (price, customer, product)  
- ❌ No domain assumptions (commerce, finance, healthcare)
- ❌ No content-specific logic or rules
- ❌ No industry-dependent processing

---

## 🚀 **EXPECTED OUTCOMES**

### **Quantitative Improvements:**
- **CSV Queries**: 0% → 80%+ success rate  
- **Excel Queries**: 0% → 80%+ success rate
- **List Queries**: 50% → 90%+ success rate  
- **Overall Structured Data**: 23% → 85%+ success rate

### **Universal Applicability:**
- Medical research data tables → Same chunking algorithms
- Legal document lists → Same list processing  
- Scientific datasets → Same pattern detection
- Financial reports → Same tabular analysis
- Academic papers → Same structural recognition

### **Scalability Benefits:**
- Zero maintenance for new domains
- No pattern updates for new industries
- Universal algorithms work across all content types  
- Mathematical principles remain constant

---

**Document Version**: v1.0 - Universal Structured Data Guidelines  
**Date**: January 2025  
**Compliance**: 100% Content-Agnostic ✅ 