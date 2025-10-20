import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.papernu_loader import _split_subject_catalog_number, _make_course_records_from_data
from ingestion.models import FullCourseRecord
from pprint import pprint

def test_split_subject_catalog_number():
    assert _split_subject_catalog_number("AF_AM_ST 327-0") == ("AF_AM_ST", "327-0")
    assert _split_subject_catalog_number("COMP_ENG 329-0") == ("COMP_ENG","329-0")
    assert _split_subject_catalog_number("COMP_SCI 301-0") == ("COMP_SCI","301-0")

def test_make_course_records_from_data():
    dummy_data = {
        "courses": [
            {
                "i": "AF_AM_ST 101-6",
                "n": "First-Year Seminar",
                "u": "1.00",
                "r": True,
                "t": [
                    "4800",
                    "4810",
                    "4850",
                    "4860",
                    "4880",
                    "4890",
                    "4900"
                ],
                "s": None,
                "f": None,
                "c": "WCAS",
                "o": [
                    [
                        "Black Women's Fiction",
                        [
                            "4800",
                            "4900"
                        ]
                    ],
                    [
                        "Black Life. Trans Life.",
                        [
                            "4810",
                            "4850",
                            "4880"
                        ]
                    ],
                    [
                        "From Black Power to Black Lives Matter",
                        [
                            "4850",
                            "4890"
                        ]
                    ],
                    [
                        "Black Gothic",
                        [
                            "4860"
                        ]
                    ],
                    [
                        "Passing and the Performance of Identity",
                        [
                            "4880"
                        ]
                    ],
                    [
                        "A Dark Rock Surged Upon': Navigating Race, Class,",
                        [
                            "4880"
                        ]
                    ],
                    [
                        "Black Creativity in the Digital Age",
                        [
                            "4900"
                        ]
                    ]
                ]
            },
            {
                "i": "AF_AM_ST 210-0",
                "n": "Survey of African American Literature",
                "u": "1.00",
                "r": False,
                "d": "Literature of Black people in the United States from slavery to freedom. Works of major writers and significant but unsung bards of the past. ENGLISH 266-0 and AF_AM_ST 210-0 are taught together.",
                "s": "6",
                "t": [
                    "4800",
                    "4850"
                ],
                "f": None,
                "c": "WCAS"
            },
            {
                "i": "AF_AM_ST 210-CN",
                "n": "Survey of African American Literature",
                "u": "1.00",
                "r": False,
                "d": "Literature of blacks in the United States from slavery to freedom. Works of major writers and significant but unsung bards of the past."
            },
            {
                "i": "AF_AM_ST 211-0",
                "n": "Literatures of the Black World",
                "u": "1.00",
                "r": False,
                "d": "Introductory survey of fiction, poetry, drama, folktales, and other literary forms of Africa and the African diaspora. Texts may span the precolonial, colonial, and postcolonial periods and cover central themes, such as memory, trauma, spirituality, struggle, identity, freedom, and humor.",
                "s": "6",
                "t": [
                    "4810",
                    "4850",
                    "4890"
                ],
                "f": None,
                "c": "WCAS",
                "o": [
                    [
                        "Black Classicism",
                        [
                            "4850"
                        ]
                    ]
                ]
            },
            {
                "i": "COMP_SCI 330-0",
                "n": "Human Computer Interaction",
                "u": "1.00",
                "r": False,
                "d": "Introduction to human-computer interaction and design of systems that work for people and their organizations. Understanding the manner in which humans interact with and use computers for productive work.",
                "p": "COMP_SCI 211-0 or Graduate standing or Consent of instructor",
                "t": [
                    "4800",
                    "4840",
                    "4880",
                    "4910",
                    "4920",
                    "4950",
                    "4960",
                    "4990",
                    "5000"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
            {
                "i": "COMP_SCI 331-0",
                "n": "Introduction to Computational Photography",
                "u": "1.00",
                "r": False,
                "d": "Fundamentals of digital imaging and modern camera architectures. Hands-on experience acquiring, characterizing, and manipulating data captured using a modern camera platform.",
                "p": "COMP_SCI 150 or COMP_SCI 211 or Consent of Instructor",
                "t": [
                    "4800",
                    "4830",
                    "4850",
                    "4950"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
            {
                "i": "COMP_SCI 333-0",
                "n": "Interactive Information Visualization",
                "u": "1.00",
                "r": False,
                "d": "This course covers theory and techniques for information visualization: the use of interactive interfaces to visualize abstract data. The course targets students interested in using visualization in their work or in building better visualization tools and systems. Students will learn to design and implement effective visualizations, critique others' visualizations, conduct exploratory visual analysis, and navigate research on information visualization.",
                "p": "COMP_SCI 214-0 or consent of instructor",
                "t": [
                    "4850",
                    "4880",
                    "4920",
                    "4990",
                    "5000"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
            {
                "i": "COMP_SCI 335-0",
                "n": "Introduction to the Theory of Computation",
                "u": "1.00",
                "r": False,
                "d": "Mathematical foundations of computation, including computability, relationships of time and space, and the P vs. NP problem.",
                "p": "COMP_SCI 212-0 or consent of instructor",
                "t": [
                    "4800",
                    "4860",
                    "4880",
                    "4920",
                    "4960",
                    "4980",
                    "5000"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
            {
                "i": "COMP_SCI 336-0",
                "n": "Design & Analysis of Algorithms",
                "u": "1.00",
                "r": False,
                "d": "Analysis techniques: solving recurrence equations. Algorithm design techniques: divide and conquer, the greedy method, backtracking, branch-and-bound, and dynamic programming. Sorting and selection algorithms, order statistics, heaps, and priority queues.",
                "p": "COMP_SCI 111-0, COMP_SCI 212-0, or CS Graduate Standing or consent of instructor",
                "t": [
                    "4800",
                    "4810",
                    "4820",
                    "4840",
                    "4850",
                    "4860",
                    "4880",
                    "4890",
                    "4900",
                    "4920",
                    "4930",
                    "4940",
                    "4960",
                    "4970",
                    "4980",
                    "4990",
                    "5000"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
            {
                "i": "COMP_ENG 203-0",
                "n": "Introduction to Computer Engineering",
                "u": "1.00",
                "r": False,
                "d": "Overview of computer engineering design. Number systems and Boolean algebra. CMOS and logic gates. Design of combinational circuits and simplification. Decoders, multiplexers, adders. Sequential logic and flip flops. Introduction to assembly language.",
                "t": [
                    "4800",
                    "4810",
                    "4820",
                    "4840",
                    "4850",
                    "4880",
                    "4900",
                    "4920",
                    "4930",
                    "4960",
                    "4970",
                    "5000"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
            {
                "i": "COMP_ENG 205-0",
                "n": "Fundamentals of Computer System Software",
                "u": "1.00",
                "r": False,
                "d": "Basics of assembly language programming. Macros. System stack and procedure calls. Techniques for writing assembly language programs. Features of Intel x86 architecture. Interfaces between C and assembly codes.",
                "p": "COMP_SCI 111-0 or GEN_ENG 205-1; COMP_ENG 203-0 recommended",
                "t": [
                    "4810",
                    "4850",
                    "4890",
                    "4930"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
            {
                "i": "COMP_ENG 295-0",
                "n": "Special Topics in Computer Engineering",
                "u": "1.00",
                "r": True,
                "d": "Topics suggested by students or faculty and approved by the department."
            },
            {
                "i": "COMP_ENG 303-0",
                "n": "Advanced Digital Design",
                "u": "1.00",
                "r": False,
                "d": "Overview of digital logic design. Technology review. Delays, timing in combinational and sequential circuits, CAD tools, arithmetic units such as ALUs and multipliers. Introduction to VHDL.",
                "p": "COMP_ENG 203-0",
                "t": [
                    "4800",
                    "4820",
                    "4840",
                    "4860",
                    "4880",
                    "4900",
                    "4920",
                    "4940",
                    "4960",
                    "4980",
                    "5000"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
            {
                "i": "COMP_ENG 329-0",
                "n": "The Art of Multicore Concurrent Programming",
                "u": "1.00",
                "r": False,
                "d": "Concurrency disciplines and practical programming techniques for multicore processors; synchronization primitives, mutual exclusion, foundation of shared memory, locking, non-blocking synchronization, and transactional memory.",
                "p": "COMP_SCI 110-0 or COMP_SCI 111-0",
                "t": [
                    "4820",
                    "4930",
                    "4970"
                ],
                "s": None,
                "f": None,
                "c": "MEAS"
            },
        ]
    }
    data: list[FullCourseRecord] = _make_course_records_from_data(dummy_data)
    for course in data:
        llm_msg = course.get_message()
        assert llm_msg != ""
        assert "Course information for" in llm_msg    

def test_basic() -> None:
    assert 4 == 4

def test_find_catalog_number_matches_course():
    from server.rag.pipeline import RAGPipeline
    from ingestion.models import FullCourseRecord

    # Create a FullCourseRecord for CS336 without calling RAGPipelinePaperNU.__init__
    record = FullCourseRecord(
        subject="COMP_SCI",
        catalog_number="336-0",
        name="Design & Analysis of Algorithms",
        description="",
        prereqs=""
    )

    # Bypass __init__ to avoid external client initialization
    pipeline = object.__new__(RAGPipeline)
    pipeline.course_records = [record]

    query = "Hey, can you tell me about CS336 and its prereqs?"
    matches = pipeline._match_catalog_number_to_course(query)

    assert any("Design & Analysis of Algorithms" in m for m in matches), f"expected course name in matches, got: {matches}"