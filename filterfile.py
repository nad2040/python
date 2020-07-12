def filterFile(oldFile, newFile):
    f1 = open(oldFile, "r")
    f2 = open(newFile, "a")
    while True:
        text = f1.readline()
        if text == "":
            break
        if text[0] == "#":
            continue
        f2.write(text)
    f1.close()
    x, y, z = 5, 10, 22.33
    f2.write (str(x) + "\n")
    f2.write ("hello %d world %f\n" % (y, z))
    f2.write ("hello {} world {}\n".format(y, z))
    f2.close()
    return

filterFile("test.dat", "tmp/test2.dat")