#PPQQ + PPQQ  = RQSPP

#P Q R S are from 0 to 9
#PPQQ are base 10 numbers
# 
# for p in range(10):
#     for q in range(10):
#         for r in range(10):
#             for s in range(10):
#                 ppqq = p * 1000 + p * 100 + q * 10 + q
#                 rqspp = r * 10000 + q * 1000 + s * 100 + p * 10 + p
#                 if ppqq + ppqq == rqspp:
#                     print "%s + %s = %s" % (ppqq, ppqq, rqspp)
# 
#straw x 4 = warts

for s in range(10):
    for t in range(10):
        for r in range(10):
            for a in range(10):
                for w in range(10):
                    straw = s * 10000 + t * 1000 + r * 100 + a * 10 + w
                    warts = w * 10000 + a * 1000 + r * 100 + t * 10 + s
                    if straw * 4 == warts:
                        print("%s * %d = %s" % (straw, 4, warts))