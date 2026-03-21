import os

def print_clean_tree(startpath):
    ignore_dirs = {'node_modules', 'venv', '__pycache__', '.git', 'data', 'global_test_data'}
    
    print("\n📂 PROJECT STRUCTURE")
    print("==================================================")
    
    for root, dirs, files in os.walk(startpath):
        # Filter out the folders we don't want to see
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('hospital_')]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}📁 {os.path.basename(root) or '.'}/")
        
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            # Hide the massive datasets and compiled files
            if not f.endswith('.npy') and not f.endswith('.pyc'):
                print(f"{subindent}📄 {f}")
                
    print("==================================================\n")

if __name__ == '__main__':
    print_clean_tree('.')