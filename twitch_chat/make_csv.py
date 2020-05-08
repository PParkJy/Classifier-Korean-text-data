import csv

#In windows, csv.writerow() has an issue that add the empty line behind each line. So, set the newline="" option.
f = open('test.csv', 'w', encoding='utf-8', newline="")
wr = csv.writer(f)
wr.writerow([1, "Alice", True])
wr.writerow([2, "Bob", False])
f.close()