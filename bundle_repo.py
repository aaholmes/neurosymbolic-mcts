import os

# Configuration
OUTPUT_FILE = "full_repository_context.txt"
# Folders to completely ignore
IGNORE_DIRS = {
    ".git", "target", "__pycache__", "venv", "env", 
    "node_modules", ".idea", ".vscode", "data", "logs"
}
# File extensions to include (add more if needed)
INCLUDE_EXTS = {
    ".rs", ".py", ".toml", ".md", ".txt", ".json", ".sh", ".yaml", ".yml"
}
# Specific files to exclude
IGNORE_FILES = {"Cargo.lock", "full_repository_context.txt", "bundle_repo.py"}

def bundle_project():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # Write a header
        outfile.write(f"PROJECT LOGOS REPOSITORY DUMP\n")
        outfile.write("==================================================\n\n")

        for root, dirs, files in os.walk("."):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                _, ext = os.path.splitext(file)
                if ext in INCLUDE_EXTS or file == "Dockerfile":
                    file_path = os.path.join(root, file)
                    
                    # Write the file separator and name
                    outfile.write(f"\n\n--- START OF FILE: {file_path} ---\n")
                    outfile.write("```\n")
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}")
                        
                    outfile.write("\n```\n")
                    outfile.write(f"--- END OF FILE: {file_path} ---\n")
                    print(f"Added: {file_path}")

    print(f"\nSuccess! All code bundled into: {OUTPUT_FILE}")

if __name__ == "__main__":
    bundle_project()
