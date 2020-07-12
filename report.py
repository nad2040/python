def report (wages) :
    students = wages.keys()
    #students.sort()
    for student in students :
        print("%-20s %12.2f" % (student, wages[student]))

wages = {'mary': 6.23, 'joe': 5.45, 'joshua': 4.25}
report (wages)