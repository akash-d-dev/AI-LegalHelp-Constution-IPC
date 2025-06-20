import re
import json

def test_ipc_extraction():
    """Test IPC extraction patterns on sample text"""
    
    # Sample text from the file
    sample_text = """The State Of Rajasthan And Anr. vs Badri And Anr. on 16 July, 1974 19. Before Criminal Amendment Act No. 25 of 1955, deathsentencewas the rule for the offence ofmurderand transportation for life an exception. If the lesserpenalty was to be imposed, then Sub-section(5) of Section 367 of the Criminal Procedure Coderequired reasons to be given. By the amendment of 1955 the Parliament seems to have taken note of the current penological thought ,and had recast the provisions of the law which has done away with the requirements of giving reasons for awarding lesser punishment. The Courthas now a discretion to award either of the twopenalties prescribed under Section 302, Indian Penal Code. According to the Supreme Courtin Chawla v. State of Haryanathe deathsentenceis now exacted where themurderwas perpetrated with marked brutality. The position is reversed with the passing of the new Codeof 1973 where Section 354(3)of the Criminal Procedure Coderequires Special reasons to be given for awardingsentenceof death. This has made the deathsentenceas an exception and life imprisonment a rule. We are left in dark about the reasons for committing themurderof Mst. Somoti. Themurderhas been committed with the aid of formidable weapon which has put an end to the life of Mst. Somoti without inflicting any torture to her. In these circumstances, we cannot hold that themurderwas a brutal one which calls for the extremepenalty of death, In view of these circumstances, while maintaining the conviction of Badri under Sections 302 and 307, Indian Penal Code, we think it proper to reduce the deathpenalty to one of life imprisonment under Section 302, Indian Penal Code. Thesentenceawarded under Section 307 Indian Penal Codeis, however, maintained. The appeal of Badri is accepted to this extent."""
    
    print("Sample text contains these obvious IPC references:")
    print("- Section 367 (Criminal Procedure Code)")
    print("- Section 302, Indian Penal Code")
    print("- Section 354(3) (Criminal Procedure Code)")
    print("- Sections 302 and 307, Indian Penal Code")
    print("- Section 302, Indian Penal Code")
    print("- Section 307 Indian Penal Code")
    print()
    
    # Test different patterns
    patterns = [
        (r'Section\s+(\d+)\s*(?:of\s+)?(?:the\s+)?(?:Indian\s+Penal\s+Code|IPC)', "Standard IPC pattern"),
        (r'Section\s+(\d+)\s*IPC', "Section X IPC"),
        (r'IPC\s+Section\s+(\d+)', "IPC Section X"),
        (r'(\d+)\s*IPC\s+Section', "X IPC Section"),
        (r'under\s+Section\s+(\d+)\s*IPC', "under Section X IPC"),
        (r'Section\s+(\d+)', "Section X (basic)"),
        (r'(\d+)\s*of\s*IPC', "X of IPC"),
        (r'IPC\s+(\d+)', "IPC X"),
        (r'(\d+)\s*IPC', "X IPC"),
        (r'Section\s+(\d+)\s*of\s*the\s*Indian\s*Penal\s*Code', "Section X of the Indian Penal Code"),
        (r'Sections\s+(\d+)\s*and\s*(\d+)', "Sections X and Y"),
        (r'Sections\s+(\d+),\s*(\d+)', "Sections X, Y"),
        (r'Section\s+(\d+)\s*and\s*(\d+)', "Section X and Y"),
        (r'Section\s+(\d+),\s*(\d+)', "Section X, Y"),
        (r'(\d+)\s*and\s*(\d+)\s*IPC', "X and Y IPC"),
        (r'(\d+),\s*(\d+)\s*IPC', "X, Y IPC"),
        (r'convicted\s+under\s+Section\s+(\d+)', "convicted under Section X"),
        (r'punished\s+under\s+Section\s+(\d+)', "punished under Section X"),
        (r'charged\s+under\s+Section\s+(\d+)', "charged under Section X"),
        (r'offence\s+under\s+Section\s+(\d+)', "offence under Section X"),
        (r'crime\s+under\s+Section\s+(\d+)', "crime under Section X"),
        (r'Sec\.\s*(\d+)', "Sec. X"),
        (r'Sec\s+(\d+)', "Sec X"),
        (r'(\d+)\s*Sec\.', "X Sec."),
        (r'(\d+)\s*Sec\s', "X Sec"),
        (r'Indian\s+Penal\s+Code\s+Section\s+(\d+)', "Indian Penal Code Section X"),
        (r'Penal\s+Code\s+Section\s+(\d+)', "Penal Code Section X"),
        (r'Code\s+Section\s+(\d+)', "Code Section X"),
        # New patterns for spaced text
        (r'Section\s+(\d+),\s*Indian\s+Penal\s+Code', "Section X, Indian Penal Code"),
        (r'Section\s+(\d+)\s*Indian\s+Penal\s+Code', "Section X Indian Penal Code"),
        (r'(\d+)\s*Indian\s+Penal\s+Code', "X Indian Penal Code"),
        (r'(\d+)\s*I\s*P\s*C', "X I P C (spaced)"),
        (r'I\s*P\s*C\s*(\d+)', "I P C X (spaced)"),
        (r'Section\s+(\d+)\s*I\s*P\s*C', "Section X I P C (spaced)"),
        (r'I\s*P\s*C\s*Section\s+(\d+)', "I P C Section X (spaced)"),
    ]
    
    found_sections = set()
    
    print("Testing patterns:")
    print("=" * 80)
    
    for pattern, description in patterns:
        matches = re.findall(pattern, sample_text, re.IGNORECASE)
        if matches:
            print(f"✓ {description}: {matches}")
            for match in matches:
                if isinstance(match, tuple):
                    for section_num in match:
                        section_num = str(section_num).strip()
                        if section_num.isdigit() and 1 <= int(section_num) <= 511:
                            found_sections.add(section_num)
                else:
                    section_num = str(match).strip()
                    if section_num.isdigit() and 1 <= int(section_num) <= 511:
                        found_sections.add(section_num)
        else:
            print(f"✗ {description}: No matches")
    
    print("\n" + "=" * 80)
    print(f"Total unique IPC sections found: {len(found_sections)}")
    print(f"IPC sections: {sorted(list(found_sections))}")
    
    # Test specific problematic patterns
    print("\nTesting specific problematic text:")
    problematic_texts = [
        "Section 302, Indian Penal Code",
        "Section 302 Indian Penal Code", 
        "Section 302 I P C",
        "302 I P C",
        "I P C 302",
        "Sections 302 and 307, Indian Penal Code"
    ]
    
    for text in problematic_texts:
        print(f"\nTesting: '{text}'")
        for pattern, description in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                print(f"  ✓ {description}: {matches}")

if __name__ == "__main__":
    test_ipc_extraction() 