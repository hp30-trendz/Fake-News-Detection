def count_lines(filename):
    with open(filename, "r") as f:
        return len(f.readlines())
    
print(count_lines("sample.txt"))