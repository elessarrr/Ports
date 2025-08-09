# Plan to Safely Remove the 'Weather Impact' Feature

## Context
The 'Weather Impact' feature is currently buggy and not essential for the core product. The goal is to remove it conservatively, ensuring no disruption to other working features and making rollback easy if needed. All new logic should be isolated, and unnecessary code/files should be avoided.

## Step-by-Step Plan

### 1. Locate All Weather Impact Code
- Search the codebase for all references to the 'Weather Impact' feature (UI, backend, data loading, etc.).
- Document all files and functions/classes that reference or implement this feature.
- Double-check for any related configuration, environment variables, or documentation.

### 2. Review Dependencies and Data Flow
- Identify any dependencies or data flows that interact with the weather feature (e.g., shared data loaders, dashboard sections, or API calls).
- Note any shared logic that must be preserved for other features.

### 3. Plan for Conservative Removal
- For each identified file/function:
  - Decide if it should be deleted, commented out, or replaced with a stub (e.g., returning N/A or hiding the UI section).
  - Prefer commenting out or stubbing code over deletion for easy rollback.
- Ensure that any UI elements (cards, charts, tabs) related to weather are removed or hidden, not just left broken.

### 4. Isolate and Backup Changes
- Before making changes, create a backup branch (e.g., `remove-weather-impact-backup`).
- For each change, add clear comments (e.g., `# Weather Impact feature removed - see planning/remove_weather_impact_plan.md`).
- If new helper functions or modules are needed for stubbing, check if similar code exists before creating anything new.

### 5. Test After Each Change
- After each removal or stub, run the application and all tests to confirm nothing else is broken.
- Check especially for:
  - Dashboard rendering (no blank or broken sections)
  - Data loading and processing
  - Any error logs or warnings

### 6. Clean Up and Document
- Once confirmed working, remove any now-unused imports, variables, or files.
- Update documentation to reflect the removal (README, data_sources.md, etc.).
- Add a summary of what was removed and why to `learnings_from_debugging.md`.

### 7. Rollback Plan
- If any issues arise, revert to the backup branch or uncomment stubbed code.
- Keep all removal changes in isolated commits for easy rollback.

## Additional Notes for Developers
- Do not remove or modify unrelated code.
- Avoid deleting files unless absolutely certain they are only used for weather.
- Ask for a code review before merging changes to main.
- If unsure about any dependency, ask for clarification before proceeding.

---

**Follow this plan step by step. If you encounter anything unexpected, pause and document it before proceeding.**