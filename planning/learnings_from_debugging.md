

# How to document


## AI IDE Debug Summary Logging Prompt

### High-Level Instruction  
Every time you fix a coding error, document the debugging process and key learnings in a maintained, recruiter-friendly summary file.

### Functional Requirements & Step-by-Step Instructions

**1. Error Fixing and Documentation**  
- After fixing a coding error, generate a concise summary including:  
  - **What caused the error** (root cause analysis)  
  - **How did we fix it** (solution/approach taken)  
  - **Learnings or takeaways** (best practices, future prevention)  
- Write clearly and professionally—a tone suitable for LinkedIn readers (e.g., hiring managers, recruiters)—emphasizing:  
  - Problem-solving skills  
  - Communication skills  
  - Evidence of learning and adaptability  

**2. Managing the Summary File**  
- Use a single markdown file: `learnings_from_debugging.md`  
- If the file doesn’t exist, create it at the root of the project/workspace.

**3. Avoiding Duplicate Learnings**  
- Before adding a new entry:  
  - Scan existing entries for similar errors or learnings.  
    - If a similar case exists, append a new **“Additional Insight”** or **“Further Example”** subsection.  
    - If the error/learning is new, create a new section with the provided template.

**4. Entry Format in Markdown**  
- Each entry should include:  
  - **Error Title** (brief, descriptive)  
  - **Caused by:** (root issue description)  
  - **How fixed:** (resolution steps or principle)  
  - **Learnings/Takeaway:** (prevention or general insight)  
  - **Date fixed:** (automatically populated with current date and time)
  - **Additional Insight (Optional):** (if updating/extending an existing entry)

- **Instruct the IDE or script to**:  
  - **Automatically query the current system date & time** immediately before adding the summary, and populate the "Date fixed" field in `YYYY-MM-DD HH:MM` format.

**5. Optimizing for LinkedIn Audience**  
- Highlight:  
  - Problem-solving process  
  - Clear, concise explanations  
  - Demonstrable skills growth

**6. End-to-End Workflow**  
- Fix Error → Summarize (using template) → Scan/Add/Update `learnings_from_debugging.md` → Automate system date/time → Save

### **Markdown Entry Template Example**

```markdown
## [Error Title Here]

- **Caused by:** Concise root cause explanation.
- **How fixed:** Key steps taken to resolve the error.
- **Learnings/Takeaway:** Key prevention tip or insight for the future.
- **Date fixed:** 2025-07-26 20:00

**Additional Insight:**  
- (If applicable, add further examples, nuances, or references to similar cases.)
```

### **Implementation Guidance**

- **Consistent Timestamping:** Using system date/time for “Date fixed” improves the reliability and professionalism of your log.
    - *Sample for Python*: `from datetime import datetime; datetime.now().strftime('%Y-%m-%d %H:%M')`
- **Why this process?**: It demonstrates to recruiters and hiring managers your concrete problem-solving, documentation skills, and commitment to continuous improvement.  
- **Tip:** Stay concise but informative, and use the same structure for every log entry.

#### Let me know if you need code samples for date/time retrieval in a specific language or want handy snippets/templates for automation!

---

# Debugging Learnings

## Port Cargo Statistics Data Structure Mismatch Error

- **Caused by:** Mismatch between expected data structure in Streamlit UI code and actual data structure returned by the `get_cargo_breakdown_analysis()` function. The UI was looking for a `data_quality` key at the top level, but the function returned it nested within `data_summary`.
- **How fixed:** Systematically analyzed the data loader function structure, identified all mismatched field names, and updated the Streamlit application to align with the actual returned data structure. Also resolved missing Python dependencies (plotly, simpy) in the pipx streamlit environment.
- **Learnings/Takeaway:** Always verify data structure contracts between data processing functions and UI components. Use consistent naming conventions and document expected data formats. When working with pipx-installed tools, ensure all dependencies are injected into the correct virtual environment.
- **Date fixed:** 2025-01-06

**Additional Insight:**
- This type of error is common in data dashboard applications where multiple developers work on backend data processing and frontend visualization separately.
- Implementing data validation or schema checking could prevent such mismatches in production.
- Environment dependency issues can be avoided by using proper dependency management and testing in the target deployment environment.

## ModuleNotFoundError for plotly in pipx Environment

- **Caused by:** Streamlit was installed via pipx, which creates an isolated virtual environment. When the application tried to import plotly, it wasn't available in the pipx streamlit environment, even though it was installed in the system Python environment.
- **How fixed:** Used `pipx inject streamlit plotly` to install plotly directly into the pipx streamlit environment, ensuring the dependency was available where Streamlit was running.
- **Learnings/Takeaway:** When using pipx for tool installation, remember that each tool runs in its own isolated environment. Dependencies must be injected into the specific pipx environment using `pipx inject <tool> <dependency>` rather than installing globally.
- **Date fixed:** 2025-01-06

## Conda Environment Path Resolution Error

- **Caused by:** Attempted to run Streamlit using `conda run -n TraeAI-8` but the environment path was incorrect or the environment didn't exist, resulting in "EnvironmentLocationNotFound" error.
- **How fixed:** Used `conda env list` to identify available environments, discovered the current environment was `TraeAI-9`, and switched to using the pipx approach instead of conda for this specific case.
- **Learnings/Takeaway:** Always verify environment names and paths before attempting to activate them. Use `conda env list` or `conda info --envs` to check available environments. Consider environment management strategy consistency across the project.
- **Date fixed:** 2025-01-06

## Missing simpy Dependency Error

- **Caused by:** The port simulation module (`src/core/port_simulation.py`) required the `simpy` library for discrete event simulation, but it wasn't installed in the pipx streamlit environment.
- **How fixed:** Installed simpy into the pipx streamlit environment using `pipx inject streamlit simpy` and restarted the Streamlit application to ensure the new dependency was loaded.
- **Learnings/Takeaway:** When working with specialized libraries (like simulation frameworks), ensure all project dependencies are properly installed in the target runtime environment. Always check requirements.txt for complete dependency lists and install them systematically.
- **Date fixed:** 2025-01-06

## CSV File Loading and Data Processing Errors

- **Caused by:** Multiple issues with CSV file processing including file path resolution, data type mismatches, and missing error handling for malformed data files.
- **How fixed:** Implemented robust error handling in data_loader.py, added file existence checks, improved data type conversion with fallbacks, and added informative error messages for debugging.
- **Learnings/Takeaway:** Always implement comprehensive error handling for file I/O operations. Use try-catch blocks with specific exception handling for different failure modes (file not found, parsing errors, data type issues). Provide clear error messages that help identify the root cause.
- **Date fixed:** 2025-01-06
