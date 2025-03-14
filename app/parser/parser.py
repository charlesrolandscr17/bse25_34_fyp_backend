from google import genai

from extract_text import extract_text_from_docx

path = "Data/Resume_template.docx"

docx_text = extract_text_from_docx(path)

format = """
"input": {
        "job_description": "### Business Development Executive\n\n**Company:** Innovate Dynamics Ltd.\n\n**Location:** Hyderabad, India\n\n**Role Overview:**\nJoin Innovate Dynamics Ltd. as a Business Development Executive, where you will play a pivotal role in driving business growth by generating leads and managing key client accounts. Leverage your expertise in economic analysis and marketing strategies to enhance our market presence and build strong client relationships.\n\n**Responsibilities:**\n- Develop and execute comprehensive business development plans to achieve company objectives.\n- Identify and engage prospective clients across diverse niches, including software development and outsourcing.\n- Manage and grow relationships with key accounts to drive long-term success.\n- Utilize digital marketing strategies to boost client engagement and retention.\n- Collaborate with cross-functional teams to develop tailored client solutions.\n- Utilize LinkedIn Sales Navigator to identify and approach new business leads.\n\n**Qualifications:**\n- Bachelor\u2019s degree in Business Administration, Economics, or related field.\n- 3-5 years of experience in business development or related fields.\n- Strong proficiency with MS Office Suite and data analysis tools like Stata.\n- Excellent verbal and written communication skills.\n- Ability to multitask and manage multiple projects simultaneously.",
        "macro_dict": {
            "leadership": 32,
            "experience": 27,
            "communication skills": 41
        },
        "micro_dict": {
            "ms excel": 66,
            "ms word": 34
        },
        "additional_info": "Innovate Dynamics Ltd. prioritizes applicants from top-tier universities such as [University X] and [University Y]. Relocation assistance is available for candidates willing to move to Hyderabad. Technical skills are assessed through standardized tests and a series of interviews. Our company encourages diversity and inclusivity in the hiring process.",
        "minimum_requirements": [
            "Ability to manage multiple projects simultaneously"
        ],
        "resume": " \n \nEXPERIENCE  \nBusiness Development Manager  @FINLOYD  (2012  > Present)  \nI. Develop ing growth strategies by analyzing market strategies.  \nII. Making N egotiating strategies by integration of new venture \nwith company strategies.  \nIII. To Secure & Protect the company\u2019s confidential information.  \nIV. Managing & Retaining Relationship with existing Clien ts. \nV. Developing Sales goals for team & ensure they met.  \nVI. Making q uotes & Proposals  \n \n \n Sales Team Mentor  @ALPHA Group (2010 -2012 ) \n I. To evaluate new opportunities by quantifies  business \nintelligence.  \nII. Promotes a professional company image by excellent services, \nsales & Communication skills.  \nIII. Possess a strong knowledge of product  & our competitors in the \nindustry.  \nIV. Ability to solve the tough problems with creative talents.  \n \n Sales Executive @ALPHA Group (2009-2010 ) \n I. Planning & Organizing d aily work schedule on potential clients  \nII. Making Business Presentations according to the industry.  \nIII. Organizational skills are important in order to pri oritize tasks & \nmeet deadlines.  \nIV. To Expand the clients, teamwork skills are very much efficient.  \nEDUCATION  \nSikkim Manipal University  (2009 -2013) \nBBA- Bachelor in Business Administrations.  \nWESTECH COLLEGE, UK  (2006-2007)  \nDiploma in Business Administrations, DBA  ABOUT ME  \n1. I am always into maintaining fruitful \nrelationships with existing clients.  \n2. Ability to handle pressure & meet \ndeadlines.  \n3. In depth knowledge of security \nindustry fields.  \n4. Excellent time management & \norganizations.  \n5. Strong Proficiency in Microsoft \nword, Excel, Power Point & E -media \nCS, ZOHO, Card Presso.  \n6. Ability to communicate informations \nwith the technical client in a concise \nmanner.  \n7. Technical skills of providing the \nclients with the better & convincin g \nsolutions.  \n8. Excellent organizational skills with \nemphasis on priorities & goal \nsetting.  \n9. Experiences working to & exceeding \ntargets.  \n10. Able to provide quality leadership to \na large team of sales people.  Zain Younas  \nSenior Business development/ Business \ndevelopment Specialist/ Sales Team Leader/ Sr. \nSecurity Consultant/ Sales Specialist.  \nMob# +966544352196  ---- zainyounas@gmail.com  \nEXPER TISE \nSales  Goals  \nMarket Knowledge  \nTerritory M anagement  \nCommunication  Skills  \nManagement Skills  \nNegotiation Skills  "
    },
    "details": {
        "name": "Zain Younas",
        "number": "966544352196",
        "skills": [
            "Microsoft Word",
            "Excel",
            "PowerPoint",
            "E-media",
            "CS",
            "ZOHO",
            "Card Presso"
        ],
        "email_id": "zainyounas@gmail.com",
        "location": "",
        "projects": [],
        "education": [
            {
                "end_date": "05-2013",
                "university": "Sikkim Manipal University",
                "degree_title": "BBA- Bachelor in Business Administrations"
            },
            {
                "end_date": "05-2007",
                "university": "WESTECH COLLEGE, UK",
                "degree_title": "Diploma in Business Administrations, DBA"
            }
        ],
        "achievements": [],
        "publications": [],
        "certifications": [],
        "additional_urls": [],
        "executive_summary": "Zain Younas is a seasoned business development manager with extensive experience in sales and client relationship management. He possesses strong technical skills in various software applications and has a proven track record in developing growth strategies and achieving sales goals.",
        "employment_history": [
            {
                "details": "Developing growth strategies by analyzing market strategies, negotiating strategies, securing company\u2019s confidential information, managing client relationships, and developing sales goals for the team.",
                "end_date": "Present",
                "location": "",
                "job_title": "Business Development Manager",
                "start_date": "01-2012",
                "company_name": "FINLOYD"
            },
            {
                "details": "Evaluating new opportunities by quantifying business intelligence and promoting a professional company image through excellent services.",
                "end_date": "12-2012",
                "location": "",
                "job_title": "Sales Team Mentor",
                "start_date": "01-2010",
                "company_name": "ALPHA Group"
            },
            {
                "details": "Planning and organizing daily work schedules for potential clients and making business presentations according to the industry.",
                "end_date": "12-2010",
                "location": "",
                "job_title": "Sales Executive",
                "start_date": "01-2009",
                "company_name": "ALPHA Group"
            }
        ]
    }
"""

client = genai.Client(api_key="[API-Key]")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=f"""
    Can you extract data from this resume. Your response should be in json format. use this format as you extract the data from the resume.
    {format};
    {docx_text}
    """,
)

with open("Data/extracted_data.md", "w") as f:
    f.write(response.text)
