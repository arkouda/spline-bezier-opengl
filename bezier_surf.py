# coding=utf-8
import pygame
import random
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import PIL.Image as p
import numpy as np
import math as m


def plotAxis():
    glBegin(GL_LINES)
    glVertex3f(-100, 0, 0)
    glVertex3f(100, 0, 0)
    glEnd()
    glBegin(GL_LINES)
    glVertex3f(0, 100, 0)
    glVertex3f(0, -100, 0)
    glEnd()
    glBegin(GL_LINES)
    glVertex3f(0, 0, -100)
    glVertex3f(0, 0, 100)
    glEnd()


mn = np.matrix([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])


def listToVMat(ls):
    return list(map(lambda x: [x], ls))


def getXYZ(val, pXYZ):
    # print(list(map(lambda x: list(map(lambda y: y[val], x)), pXYZ)))
    return list(map(lambda x: list(map(lambda y: y[val], x)), pXYZ))


def genVMat4x1(value):
    vMat = list(map(lambda x: [x], [value ** x for x in [3, 2, 1, 0]]))
    vMat = np.matrix(vMat)
    return vMat


def genHMat1x4(value):
    hMat = [[value ** x for x in [3, 2, 1, 0]]]
    hMat = np.matrix(hMat)
    return hMat


def bezierSurface4x4(pXYZ, u, v):
    pX = np.matrix(getXYZ(0, pXYZ))
    pY = np.matrix(getXYZ(1, pXYZ))
    pZ = np.matrix(getXYZ(2, pXYZ))
    matU = genHMat1x4(u)
    matV = genVMat4x1(v)

    x = matU * mn * pX * mn * matV
    y = matU * mn * pY * mn * matV
    z = matU * mn * pZ * mn * matV
    pt = (x.item(0), y.item(0), z.item(0))
    return pt
    

def bezierCurveCubic(pXYZ, t):
    pX = np.matrix(getXYZ(0, pXYZ))
    pY = np.matrix(getXYZ(1, pXYZ))
    pZ = np.matrix(getXYZ(2, pXYZ))
    matH = genHMat1x4(t)

    x = matH * mn * pX
    y = matH * mn * pY
    z = matH * mn * pZ
    pt = (x.item(0), y.item(0), z.item(0))
    return pt


def genRandom3DPoint(i, j):
    pt = (i * 0.25, j * 0.25, random.uniform(0, 1))
    return pt


def mirrorAboutPoint(ptToMirror, ptAbout):
    return (ptAbout[0]-(ptToMirror[0] - ptAbout[0]),
            ptAbout[1]-(ptToMirror[1] - ptAbout[1]),
            ptAbout[2]-(ptToMirror[2] - ptAbout[2]))


def circularBezierCurve(lCP, lKP):
    return segmentedBezierCurve(len(lCP), (lKP[0], lKP[0]), (lCP[0], mirrorAboutPoint(lCP[0], lKP[0])), lCP[1:], lKP[1:])


#                           n      (first, last)(first, last)     [] of n-1         [] of n-1    


def segmentedBezierCurve(noOfCurves, endPtsTup, endCtrlPtsTup, controlPointsList, knotPointsList):
    curveList = []          # [[4*(x,y,z)]]
    for i in range(noOfCurves):
        if not i:
            curveList.append([endPtsTup[0], endCtrlPtsTup[0], controlPointsList[0], knotPointsList[0]])
        elif i == noOfCurves - 1:
            curveList.append([knotPointsList[i-1], mirrorAboutPoint(controlPointsList[i-1], knotPointsList[i-1]), endCtrlPtsTup[1], endPtsTup[1]])
        else:
            curveList.append([knotPointsList[i-1], mirrorAboutPoint(controlPointsList[i-1], knotPointsList[i-1]), controlPointsList[i], knotPointsList[i]])

    if endPtsTup[0] == endPtsTup[1]:
        cf = curveList[0][1]
        cl = curveList[-1][2]
        curveList[0][1] = cl
        curveList[-1][2] = cf
    return curveList


def drawCurveList(curveList):
    c = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0.5, 0), (0.5, 1, 0), (0, 1, 0.5)]
    c = c + c
    pointsList = []
    k = 0
    for i in curveList:
        glColor3f(c[k][0], c[k][1], c[k][2])
        glBegin(GL_POINTS)
        glVertex3fv(i[0])
        glVertex3fv(i[1])
        glVertex3fv(i[2])
        glVertex3fv(i[3])
        glEnd()
        glBegin(GL_LINE_STRIP)
        for j in range(20):
            pointsList.append(bezierCurveCubic(list(map(lambda x: [x], i)), float(j) / 19))
            glVertex3fv(pointsList[-1])            
        glEnd()
        k += 1
    return pointsList


def curveListToCPKP(curveList):
    lCP = []
    lKP = []
    for i in range(len(curveList)):
        lKP.append(curveList[i][0])
        lCP.append(curveList[i][2])
    pt = lCP.pop()
    lCP = [pt] + lCP
    return lCP, lKP


def writePoints(fName,a):
    f = open(fName,"w")
    for i in a:
        for j in i :
            for k in j:
                for s in k:
                    f.write(str(s))
                    f.write(",")
                f.write("+")
            f.write("$")
        f.write("*")
    f.close()


def readPoints(fName):
    f = open(fName,"r")
    k = f.read()
    a1 = []
    for i in k.split("*"):
        i1 = []
        for j in i.split("$"):
            j1 = []
            for k in j.split("+"):
                l = k.split(",")
                l.pop()
                l1 = map(float, l)
                j1.append(tuple(l1))
            j1.pop()
            i1.append(j1)
        i1.pop()
        a1.append(i1)
    a1.pop()
    return a1


def controlCurvesToSurfaces(controlCurves):
    sPointsMatrix = []
    cc = []
    while controlCurves[1:]:
        cc.append(controlCurves[0:4])
        controlCurves = controlCurves[3:]
    for i in range(len(cc)):
        spf = []
        for j in range(4):
            sp = []
            for k in range(4):
                sp.append(cc[i][k][j])
            spf.append(sp)
        sPointsMatrix.append(spf)

    surfaceMatrix = []
    for i in sPointsMatrix:
        adjacentSurfacesEW = []
        for surfacePoints in i:
            adjacentSurfacesEW.append(BSurface(surfacePoints))
        surfaceMatrix.append(adjacentSurfacesEW)

    for i in surfaceMatrix:
        for k in range(len(i)):
            if k > len(i) - 2:
                i[k].setEdge(E, i[0], True)
            else:
                i[k].setEdge(E, i[k + 1], True)

    for i in range(len(surfaceMatrix) - 1):
        for j in range(len(surfaceMatrix[i])):
            surfaceMatrix[i][j].setEdge(S, surfaceMatrix[i + 1][j], True)

    for i in surfaceMatrix:
        for j in i:
            j.evaluateSurface()

    return surfaceMatrix


####################################
#     BEZIER SURFACE GENERALIZATION
#
#            n
#          _____
#         |1 1 0|
#       w |2 5 7| e
#         |7 4 2|
#          ̅ ̅ ̅ ̅ ̅
#            s

N = 'n'
E = 'e'
S = 's'
W = 'w'

oppDict = {N: S, S: N, E: W, W: E}
adjDict = {N: (E, W), S: (E, W), E: (N, S), W: (N, S)}


def oppDirection(direction):
    return oppDict[direction]


def adjDirection(direction):
    return adjDict[direction]


class BSurface(object):
    #                     4 X 4 matrix
    def __init__(self, controlPointsMatrix):
        self.controlPointsMatrix = controlPointsMatrix
        self.edges = {N: None, E: None, S: None, W: None}
        self.texturePointsSaved = []

    def setEdge(self, edgeDirection, bSurface, dominance):
        self.edges[edgeDirection] = (bSurface, dominance)
        bSurface.edges[oppDirection(edgeDirection)] = (self, not dominance)

    def getEdgePoints(self, direction):
        if direction == N:
            return self.controlPointsMatrix[0]
        elif direction == S:
            return self.controlPointsMatrix[-1]
        elif direction == W:
            return list(map(lambda x: x[0],self.controlPointsMatrix))
        elif direction == E:
            return list(map(lambda x: x[-1],self.controlPointsMatrix))

    def getControlPoints(self, direction):
        if direction == N:
            return self.controlPointsMatrix[1]
        elif direction == S:
            return self.controlPointsMatrix[-2]
        elif direction == W:
            return list(map(lambda x: x[1], self.controlPointsMatrix))
        elif direction == E:
            return list(map(lambda x: x[-2], self.controlPointsMatrix))

    def setControlPoints(self, direction, pointsList):
        if direction == N:
            self.controlPointsMatrix[1] = pointsList
        elif direction == S:
            self.controlPointsMatrix[-2] = pointsList
        elif direction == W:
            for i in range(len(self.controlPointsMatrix)):
                self.controlPointsMatrix[i][1] = pointsList[i]
        elif direction == E:
            for i in range(len(self.controlPointsMatrix)):
                self.controlPointsMatrix[i][-2] = pointsList[i]

    def setEdgePoints(self, direction, pointsList):
        if direction == N:
            self.controlPointsMatrix[0] = pointsList
        elif direction == S:
            self.controlPointsMatrix[-1] = pointsList
        elif direction == W:
            for i in range(len(self.controlPointsMatrix)):
                self.controlPointsMatrix[i][0] = pointsList[i]
        elif direction == E:
            for i in range(len(self.controlPointsMatrix)):
                self.controlPointsMatrix[i][-1] = pointsList[i]

    def adjustPoints(self, edgeDirection, bSurface):
        edgePoints = self.getEdgePoints(edgeDirection)
        controlPoints = bSurface.getControlPoints(oppDirection(edgeDirection))
        mirroredPoints = [0 for _ in edgePoints]
        for i in range(len(edgePoints)):
            mirroredPoints[i] = mirrorAboutPoint(controlPoints[i], edgePoints[i])
        self.setControlPoints(edgeDirection, mirroredPoints)
        adj1, adj2 = adjDirection(edgeDirection)
        if self.edges[adj1]:
            self.edges[adj1][0].setEdgePoints(oppDirection(adj1), self.getEdgePoints(adj1))
        if self.edges[adj2]:
            self.edges[adj2][0].setEdgePoints(oppDirection(adj2), self.getEdgePoints(adj2))
        
    def evaluateSurface(self):
        for edgeDirection in self.edges:
            if self.edges[edgeDirection] and not self.edges[edgeDirection][1]:
                self.adjustPoints(edgeDirection, self.edges[edgeDirection][0])

    def generatePoints(self, noOfPoints):
        pointsMatrix = []
        for i in range(noOfPoints + 1):
            rowPoints = []
            for j in range(noOfPoints + 1):
                rowPoints.append(bezierSurface4x4(self.controlPointsMatrix, float(i) / noOfPoints, float(j) / noOfPoints))
            pointsMatrix.append(rowPoints)
        return pointsMatrix

    def drawQuads(self, n):
        pointsMatrix = self.generatePoints(n)
        for i in range(len(pointsMatrix) - 1):
            for j in range(len(pointsMatrix[0]) - 1):
                glBegin(GL_QUADS)
                # glTexCoord2f(0.0, 0.0)
                glTexCoord2f(abs(pointsMatrix[i][j][0]), abs(pointsMatrix[i][j][1]))
                glVertex3fv(pointsMatrix[i][j])
                # glTexCoord2f(1.0, 0.0)
                glTexCoord2f(abs(pointsMatrix[i + 1][j][0]), abs(pointsMatrix[i + 1][j][1]))
                glVertex3fv(pointsMatrix[i + 1][j])
                # glTexCoord2f(1.0, 1.0)
                glTexCoord2f(abs(pointsMatrix[i + 1][j + 1][0]), abs(pointsMatrix[i + 1][j + 1][1]))
                glVertex3fv(pointsMatrix[i + 1][j + 1])
                # glTexCoord2f(0.0, 1.0)
                glTexCoord2f(abs(pointsMatrix[i][j + 1][0]), abs(pointsMatrix[i][j + 1][1]))
                glVertex3fv(pointsMatrix[i][j+1])
                glEnd()
        return

    def drawTrigs(self, n):
        pointsMatrix = self.generatePoints(n)
        if not self.texturePointsSaved:
            for i in range(len(pointsMatrix) - 1):
                tps = []
                for j in range(len(pointsMatrix[0]) - 1):
                    tps.append([(abs(pointsMatrix[i][j][0]), abs(pointsMatrix[i][j][1])),
                                                    (abs(pointsMatrix[i + 1][j][0]), abs(pointsMatrix[i + 1][j][1])),
                                                    (abs(pointsMatrix[i][j + 1][0]), abs(pointsMatrix[i][j + 1][1])),
                                                    (abs(pointsMatrix[i + 1][j + 1][0]),abs(pointsMatrix[i + 1][j + 1][1]))])
                    glBegin(GL_TRIANGLE_STRIP)
                    # glTexCoord2f(0.0, 0.0)
                    glTexCoord2f(abs(pointsMatrix[i][j][0]), abs(pointsMatrix[i][j][1]))
                    glVertex3fv(pointsMatrix[i][j])
                    # glTexCoord2f(1.0, 0.0)
                    glTexCoord2f(abs(pointsMatrix[i + 1][j][0]), abs(pointsMatrix[i + 1][j][1]))
                    glVertex3fv(pointsMatrix[i + 1][j])
                    # glTexCoord2f(0.0, 1.0)
                    glTexCoord2f(abs(pointsMatrix[i][j + 1][0]), abs(pointsMatrix[i][j + 1][1]))
                    glVertex3fv(pointsMatrix[i][j + 1])
                    # glTexCoord2f(1.0, 1.0)
                    glTexCoord2f(abs(pointsMatrix[i + 1][j + 1][0]), abs(pointsMatrix[i + 1][j + 1][1]))
                    glVertex3fv(pointsMatrix[i + 1][j + 1])
                    glEnd()
                self.texturePointsSaved.append(tps)
        else:
            for i in range(len(pointsMatrix) - 1):
                for j in range(len(pointsMatrix[0]) - 1):
                    glBegin(GL_TRIANGLE_STRIP)
                    # glTexCoord2f(0.0, 0.0)
                    glTexCoord2f(self.texturePointsSaved[i][j][0][0], self.texturePointsSaved[i][j][0][1])
                    glVertex3fv(pointsMatrix[i][j])
                    # glTexCoord2f(1.0, 0.0)
                    glTexCoord2f(self.texturePointsSaved[i][j][1][0], self.texturePointsSaved[i][j][1][1])
                    glVertex3fv(pointsMatrix[i + 1][j])
                    # glTexCoord2f(0.0, 1.0)
                    glTexCoord2f(self.texturePointsSaved[i][j][2][0], self.texturePointsSaved[i][j][2][1])
                    glVertex3fv(pointsMatrix[i][j + 1])
                    # glTexCoord2f(1.0, 1.0)
                    glTexCoord2f(self.texturePointsSaved[i][j][3][0], self.texturePointsSaved[i][j][3][1])
                    glVertex3fv(pointsMatrix[i + 1][j + 1])

                    glEnd()
        return

    def drawPoints(self, n):
        pointsMatrix = self.generatePoints(n)
        for i in range(len(pointsMatrix) - 1):
            for j in range(len(pointsMatrix[0]) - 1):
                glBegin(GL_POINTS)
                # glTexCoord2f(0.0, 0.0)
                # glTexCoord2f(abs(pointsMatrix[i][j][0]), abs(pointsMatrix[i][j][1]))
                glVertex3fv(pointsMatrix[i][j])
                # glTexCoord2f(1.0, 0.0)
                # glTexCoord2f(abs(pointsMatrix[i + 1][j][0]), abs(pointsMatrix[i + 1][j][1]))
                glVertex3fv(pointsMatrix[i + 1][j])
                # glTexCoord2f(1.0, 1.0)
                # glTexCoord2f(abs(pointsMatrix[i + 1][j + 1][0]), abs(pointsMatrix[i + 1][j + 1][1]))
                glVertex3fv(pointsMatrix[i + 1][j + 1])
                # glTexCoord2f(0.0, 1.0)
                # glTexCoord2f(abs(pointsMatrix[i][j + 1][0]), abs(pointsMatrix[i][j + 1][1]))
                glVertex3fv(pointsMatrix[i][j+1])
                glEnd()
        return


def toTuple(npMatrix):
    return (npMatrix.item((0,0)), npMatrix.item((1,0)), npMatrix.item((2,0)))


def rotateCurveByAngle(curveCrossSection, angleInDegrees):
    angleInRadians = 0.01745329252 * angleInDegrees
    rotationMatrix = np.matrix([[m.cos(angleInRadians), m.sin(angleInRadians), 0],
                                [-m.sin(angleInRadians), m.cos(angleInRadians), 0],
                                [0, 0, 1]])
    crossSection = list(map(lambda curve: list(map(lambda (x, y, z): toTuple(rotationMatrix * np.matrix([[x], [y], [z]])), curve)), curveCrossSection))
    return crossSection


def rotateControlCurves(controlCurves, angle):
    cc = []
    a = angle/len(controlCurves)
    for i in range(len(controlCurves)):
        cc.append(rotateCurveByAngle(controlCurves[i], i*a))
    return cc


def main():
    controlCurves = readPoints('cc.txt')
    surfaceMatrix = controlCurvesToSurfaces(controlCurves)
    cpkp = list(map(lambda i: curveListToCPKP(i), controlCurves))
    mode = 'e'
    curveNo = 0
    cpSelected = True
    indexSelected = 0
    changeFLAG = False
    pygame.init()
    display = (800, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    im = p.open('skinTex3.jpg')
    ix, iy, image = im.size[0], im.size[1], im.tobytes("raw", "RGB", 0, -1)
    ID = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, ID)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    glEnable(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    glScalef(0.7, 0.7, 0.7)
    # eX, eY, eZ = 0, 0, 1.5
    # rX, rY, rZ = 0, 0, 0

    # gluLookAt(0.2, 0, 0,
    #           0, 0, 0,
    #           0, 1, 0)
    # gluPerspective(120 * 0.01745329252, 1.5, 1.5, -2)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PERIOD:
                    writePoints('cc.txt', controlCurves)
            if mode == 'd':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        mode = 'e'
                        pygame.time.delay(100)
                    if event.key == pygame.K_UP:
                        glRotatef(10, 1, 0, 0)
                    if event.key == pygame.K_DOWN:
                        glRotatef(-10, 1, 0, 0)
                    if event.key == pygame.K_RIGHT:
                        glRotatef(10, 0, 1, 0)
                    if event.key == pygame.K_LEFT:
                        glRotatef(-10, 0, 1, 0)
                    # if event.key == pygame.K_w:
                    #
                    # if event.key == pygame.K_a:
                    #
                    # if event.key == pygame.K_s:
                    #
                    # if event.key == pygame.K_d:

                    if event.key == pygame.K_f:
                        controlCurves = rotateControlCurves(controlCurves, 10)
                        surfaceMatrix = controlCurvesToSurfaces(controlCurves)
                        cpkp = list(map(lambda i: curveListToCPKP(i), controlCurves))
                    if event.key == pygame.K_g:
                        controlCurves = rotateControlCurves(controlCurves, -10)
                        surfaceMatrix = controlCurvesToSurfaces(controlCurves)
                        cpkp = list(map(lambda i: curveListToCPKP(i), controlCurves))
            else:
                if event.type == pygame.KEYDOWN:
                    changeFLAG = True
                    if event.key == pygame.K_m:
                        mode = 'd'
                        pygame.time.wait(100)
                    if event.key == pygame.K_1:
                        indexSelected = 0
                    if event.key == pygame.K_2:
                        indexSelected = 1
                    if event.key == pygame.K_3:
                        indexSelected = 2
                    if event.key == pygame.K_4:
                        indexSelected = 3
                    if event.key == pygame.K_r:
                        cpSelected = not cpSelected
                    if event.key == pygame.K_MINUS:
                        if curveNo >= len(controlCurves) - 1:
                            curveNo = 0
                        else:
                            curveNo += 1
                    if event.key == pygame.K_i:
                        cp = list(map(lambda (x, y, z): (x, y, z - 0.1), cpkp[curveNo][0]))
                        kp = list(map(lambda (x, y, z): (x, y, z - 0.1), cpkp[curveNo][1]))
                        cpkp[curveNo] = (cp, kp)
                    if event.key == pygame.K_k:
                        cp = list(map(lambda (x, y, z): (x, y, z + 0.1), cpkp[curveNo][0]))
                        kp = list(map(lambda (x, y, z): (x, y, z + 0.1), cpkp[curveNo][1]))
                        cpkp[curveNo] = (cp, kp)
                    if event.key == pygame.K_UP:
                        if cpSelected:
                            cpkp[curveNo][0][indexSelected] = (cpkp[curveNo][0][indexSelected][0], cpkp[curveNo][0][indexSelected][1] + 0.001, cpkp[curveNo][0][indexSelected][2])
                        else:
                            cpkp[curveNo][1][indexSelected] = (cpkp[curveNo][1][indexSelected][0], cpkp[curveNo][1][indexSelected][1] + 0.001, cpkp[curveNo][1][indexSelected][2])
                    if event.key == pygame.K_DOWN:
                        if cpSelected:
                            cpkp[curveNo][0][indexSelected] = (cpkp[curveNo][0][indexSelected][0], cpkp[curveNo][0][indexSelected][1] - 0.001, cpkp[curveNo][0][indexSelected][2])
                        else:
                            cpkp[curveNo][1][indexSelected] = (cpkp[curveNo][1][indexSelected][0], cpkp[curveNo][1][indexSelected][1] - 0.001, cpkp[curveNo][1][indexSelected][2])
                    if event.key == pygame.K_LEFT:
                        if cpSelected:
                            cpkp[curveNo][0][indexSelected] = (cpkp[curveNo][0][indexSelected][0] - 0.001, cpkp[curveNo][0][indexSelected][1], cpkp[curveNo][0][indexSelected][2])
                        else:
                            cpkp[curveNo][1][indexSelected] = (cpkp[curveNo][1][indexSelected][0] - 0.001, cpkp[curveNo][1][indexSelected][1], cpkp[curveNo][1][indexSelected][2])
                    if event.key == pygame.K_RIGHT:
                        if cpSelected:
                            cpkp[curveNo][0][indexSelected] = (cpkp[curveNo][0][indexSelected][0] + 0.001, cpkp[curveNo][0][indexSelected][1], cpkp[curveNo][0][indexSelected][2])
                        else:
                            cpkp[curveNo][1][indexSelected] = (cpkp[curveNo][1][indexSelected][0] + 0.001, cpkp[curveNo][1][indexSelected][1], cpkp[curveNo][1][indexSelected][2])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPointSize(3)
        if changeFLAG:
            controlCurves[curveNo] = circularBezierCurve(cpkp[curveNo][0], cpkp[curveNo][1])
            if mode == 'd':
                surfaceMatrix = controlCurvesToSurfaces(controlCurves)
            changeFLAG = False
        if mode == 'e':
            plotAxis()
            glBegin(GL_LINES)
            if cpSelected:
                glVertex3fv(cpkp[curveNo][0][indexSelected])
                x1, y1, z1 = cpkp[curveNo][0][indexSelected]
                glVertex3fv((x1, y1+0.05, z1))
            else:
                glVertex3fv(cpkp[curveNo][1][indexSelected])
                x1, y1, z1 = cpkp[curveNo][1][indexSelected]
                glVertex3fv((x1, y1+0.05, z1))
            glEnd()
            drawCurveList(controlCurves[curveNo])
        else:
            for i in surfaceMatrix:
                for j in i:
                    j.drawTrigs(10)
        pygame.display.flip()
for i in range(noOfPoints + 1):
    for j in range(noOfPoints + 1):
        pointsArray.append(bezierSurface4x4(pij, float(i)/noOfPoints, float(j)/noOfPoints))
        # print(pointsArray[-1])


def toCountDict(arr):
    d = {}
    for i in range(len(arr)):
        if arr[i] in d:
            d[arr[i]] += 1
        else:
            d[arr[i]] = 1
    return d



main()

# bSurf1 = BSurface(bCP1)
# bSurf2 = BSurface(bCP2)
# bSurf1.setEdge(S, bSurf2, True)
# bSurf1.evaluateSurface()
# bSurf2.evaluateSurface()
# ptsBS1 = bSurf1.generatePoints(15)
# ptsBS2 = bSurf2.generatePoints(15)
# print(ptsBS1 + ptsBS2)



# glBegin(GL_POINTS)
# for i in pij2:
#     for j in i:
#         glVertex3fv(j)
# glEnd()

# glBegin(GL_LINE_STRIP)
# for pt in pointsArray:
#     glVertex3fv(pt)
# glEnd()

# glBegin(GL_LINE_STRIP)
# for pt in pointsArray1:
#     glVertex3fv(pt)
# glEnd()
# segmentedClosedF(listOfKnots,listOfCP,n)
# glBegin(GL_LINE_STRIP)
# for pt in pts:
#     glVertex3fv(pt)
# glEnd()


# def mirrorCurvePoints(i):
#     j = 0
#     if i == 2:
#         j = 3
#     return [(pij2[j][0][0]-(pij2[i][0][0]-pij2[j][0][0]),
#              pij2[j][0][1]-(pij2[i][0][1]-pij2[j][0][1]),
#              pij2[j][0][2]-(pij2[i][0][2]-pij2[j][0][2]))]
#
#
# def mirrorCurvePoints1(pCP,knot):
#     return [(knot[0]-(pCP[0]- knot[0]),
#              knot[1]-(pCP[1]- knot[1]),
#              knot[2]-(pCP[2]- knot[2]))]


# pij = [[genRandom3DPoint(i, j) for j in range(4)] for i in range(4)]
#
# pointsArray = []
# pointsArray1 = []
#
# noOfPoints = 20

# pij2 = [[(-0.2, 0, 0.5)], [(-0.25, 0.3, 0.5)], [(0.25, 0.3, 0.5)], [(0.2, 0, 0.5)]]
# pij3 = [pij2[0],mirrorCurvePoints(1),mirrorCurvePoints(2),pij2[3]]

#########
# pij2 = [(-0.2, 0, 0.5),(0.2, 0, 0.5)]
# pij3 = [(-0.15,-0.3,0.5),(0.25, 0.3, 0.5)]

# list(map(lambda x:[x],[genRandom3DPoint(i, j) for j in range(4)]))
# print(pij2)

# for i in range(noOfPoints + 1):
#     #for j in range(noOfPoints + 1):
#     pointsArray.append(bezierCurveCubic(pij2, float(i)/noOfPoints)) #, float(j)/noOfPoints))
#         # print(pointsArray[-1])

# for i in range(noOfPoints + 1):
#     #for j in range(noOfPoints + 1):
#     pointsArray1.append(bezierCurveCubic(pij3, float(i)/noOfPoints)) #, float(j)/noOfPoints))
#         # print(pointsArray[-1])
########### Added ############
# n = 8


# def segmentedClosedF(lKnots,lCP,n):
#     for i in range(n):
#         pijN = [[lKnots[i-1]],mirrorCurvePoints1(lCP[i-1],lKnots[i-1]),[lCP[i]],[lKnots[i]]]
#         # print(pijN)
#         pointsArrayN = []
#
#         glBegin(GL_POINTS)
#         for i in pijN:
#             for j in i:
#                 glVertex3fv(j)
#         glEnd()
#         for i in range(noOfPoints + 1):
#             pointsArrayN.append(bezierCurveCubic(pijN, float(i)/noOfPoints))
#         # print(len(pointsArrayN))
#         glBegin(GL_LINE_STRIP)
#         for pt in pointsArrayN:
#             glVertex3fv(pt)
#         glEnd()


##############################

# listOfKnots = [(0, 0.2, 0.5), (0.1, 0.1, 0.5), (0.2, 0, 0.5), (0.1, -0.1, 0.5), (0, -0.2, 0.5), (-0.1, -0.1, 0.5), (-0.2, 0, 0.5), (-0.1, 0.1, 0.5)]
# listOfCP = [(0.15, 0.175, 0.5), (0.075, 0.125, 0.5), (0.175, 0.075, 0.5), (0.125, -0.075, 0.5), (0.025, -0.175, 0.5), (-0.075, -0.175, 0.5), (-0.175, -0.025, 0.5), (-0.175, 0.075, 0.5)]
#
# lKcircle = [(0, 0.2, 0.5), (0.2, 0, 0.5), (0, -0.2, 0.5), (-0.2, 0, 0.5)]
# lCPcircle = [(-0.1104569499662, 0.2, 0.5), (0.2, 0.1104569499662, 0.5), (0.1104569499662, -0.2, 0.5), (-0.2, -0.1104569499662, 0.5)]
#
# l1 = ([(-0.022000000000000013, 0.25100000000000006, 0.5), (0.21700000000000003, 0.02100000000000001, 0.5), (0.058000000000000045, -0.14199999999999996, 0.5), (-0.188, -0.014000000000000005, 0.5)], [(-0.2454569499662001, 0.193, 0.5), (0.23500000000000004, 0.10645694996619998, 0.5), (0.12145694996619999, -0.10199999999999992, 0.5), (-0.11799999999999994, -0.1284569499662, 0.5)])
# l2 = ([(-0.022000000000000013, 0.2770000000000001, 0.5), (0.2690000000000001, 0.06500000000000004, 0.5), (0.058000000000000045, -0.13299999999999995, 0.5), (-0.20800000000000002, 0.009000000000000001, 0.5)], [(-0.2544569499662001, 0.24600000000000005, 0.5), (0.23600000000000004, 0.17345694996620004, 0.5), (0.10645694996619998, -0.12699999999999995, 0.5), (-0.15799999999999997, -0.10945694996619998, 0.5)])
# l3 = ([(-0.022000000000000013, 0.24500000000000005, 0.5), (0.24100000000000005, 0.08500000000000006, 0.5), (0.08200000000000006, -0.0819999999999999, 0.5), (-0.181, 0.009000000000000001, 0.5)], [(-0.16545694996620003, 0.20900000000000002, 0.5), (0.23600000000000004, 0.17345694996620004, 0.5), (0.20145694996620006, -0.05199999999999988, 0.5), (-0.13199999999999995, -0.10645694996619998, 0.5)])
# l4 = ([(0.022000000000000013, 0.194, 0.5), (0.21700000000000003, 0.06800000000000005, 0.5), (0.05000000000000003, -0.045999999999999874, 0.5), (-0.15599999999999997, 0.04300000000000003, 0.5)], [(-0.09245694996619996, 0.187, 0.5), (0.20800000000000002, 0.15145694996620002, 0.5), (0.19345694996620005, -0.04099999999999987, 0.5), (-0.14299999999999996, -0.06345694996619994, 0.5)])
#
# l = ([(0.003, 0.13099999999999995, 0.5), (0.16699999999999998, 0.05900000000000004, 0.5), (0.05000000000000003, -0.06599999999999989, 0.5), (-0.12999999999999995, 0.04300000000000003, 0.5)], [(-0.09245694996619996, 0.11099999999999993, 0.5), (0.16599999999999998, 0.14945694996620001, 0.5), (0.15045694996620002, -0.05999999999999989, 0.5), (-0.10999999999999993, -0.05645694996619993, 0.5)])
#
#
# lK1 = l[0]
# lCP1 = l[1]

# cl1 = [[(-0.022, 0.251, -0.5), (0.201456949, 0.309, -0.5), (0.235, 0.10645695, -0.5), (0.217, 0.021, -0.5)],
#        [(0.217, 0.021, -0.5), (0.199, -0.06445695, -0.5), (0.12145695, -0.102, -0.5), (0.058, -0.142, -0.5)],
#        [(0.058, -0.142, -0.5), (-0.00545695, -0.182, -0.5), (-0.118, -0.12845695, -0.5), (-0.188, -0.014, -0.5)],
#        [(-0.188, -0.014, -0.5), (-0.258, 0.10045695, -0.5), (-0.24545695, 0.193, -0.5), (-0.022, 0.251, -0.5)]]
#
# cl2 = [[(-0.022, 0.277, -0.2), (0.21045695, 0.308, -0.2), (0.236, 0.17345695, -0.2), (0.269, 0.065, -0.2)],
#        [(0.269, 0.065, -0.2), (0.302, -0.04345695, -0.2), (0.10645695, -0.127, -0.2), (0.058, -0.133, -0.2)],
#        [(0.058, -0.133, -0.2), (0.00954305, -0.139, -0.2), (-0.158, -0.10945695, -0.2), (-0.208, 0.009, -0.2)],
#        [(-0.208, 0.009, -0.2), (-0.258, 0.12745695, -0.2), (-0.25445695, 0.246, -0.2), (-0.022, 0.277, -0.2)]]
#
# cl3 = [[(-0.022, 0.245, 0.4), (0.12145695, 0.281, 0.4), (0.236, 0.17345695, 0.4), (0.241, 0.085, 0.4)],
#        [(0.241, 0.085, 0.4), (0.246, -0.00345695, 0.4), (0.20145695, -0.052, 0.4), (0.082, -0.082, 0.4)],
#        [(0.082, -0.082, 0.4), (-0.03745695, -0.112, 0.4), (-0.132, -0.10645695, 0.4), (-0.181, 0.009, 0.4)],
#        [(-0.181, 0.009, 0.4), (-0.23, 0.12445695, 0.4), (-0.16545695, 0.209, 0.4), (-0.022, 0.245, 0.4)]]
#
# cl4 = [[(0.022, 0.194, 1), (0.13645695, 0.201, 1), (0.208, 0.15145695, 1), (0.217, 0.068, 1)],
#        [(0.217, 0.068, 1), (0.226, -0.01545695, 1), (0.19345695, -0.041, 1), (0.05, -0.046, 1)],
#        [(0.05, -0.046, 1), (-0.09345695, -0.051, 1), (-0.143, -0.06345695, 1), (-0.156, 0.043, 1)],
#        [(-0.156, 0.043, 1), (-0.169, 0.14945695, 1), (-0.09245695, 0.187, 1), (0.022, 0.194, 1)]]
# cl1 = [[(-3.6e-2,0.18,-0.5),(0.18745694899999998,0.238,-0.5),(0.22099999999999997,3.545695e-2,-0.5),(0.20299999999999999,-4.999999999999999e-2,-0.5)],[(0.20299999999999999,-4.999999999999999e-2,-0.5),(0.185,-0.13545695,-0.5),(0.10745695,-0.173,-0.5),(4.4000000000000004e-2,-0.21299999999999997,-0.5)],[(4.4000000000000004e-2,-0.21299999999999997,-0.5),(-1.945695e-2,-0.253,-0.5),(-0.132,-0.19945694999999997,-0.5),(-0.202,-8.499999999999999e-2,-0.5)],[(-0.202,-8.499999999999999e-2,-0.5),(-0.272,2.945695000000001e-2,-0.5),(-0.25945695,0.12200000000000001,-0.5),(-3.6e-2,0.18,-0.5)]]
# cl2 = [[(-3.6e-2,0.20600000000000002,-0.2),(0.19645695,0.237,-0.2),(0.22199999999999998,0.10245695,-0.2),(0.255,-5.9999999999999915e-3,-0.2)],[(0.255,-5.9999999999999915e-3,-0.2),(0.288,-0.11445695,-0.2),(9.245695e-2,-0.198,-0.2),(4.4000000000000004e-2,-0.20400000000000001,-0.2)],[(4.4000000000000004e-2,-0.20400000000000001,-0.2),(-4.4569499999999995e-3,-0.21000000000000002,-0.2),(-0.17200000000000001,-0.18045695,-0.2),(-0.222,-6.199999999999999e-2,-0.2)],[(-0.222,-6.199999999999999e-2,-0.2),(-0.272,5.645695000000002e-2,-0.2),(-0.26845695,0.175,-0.2),(-3.6e-2,0.20600000000000002,-0.2)]]
# cl3 = [[(-3.6e-2,0.174,0.4),(0.10745695,0.21000000000000002,0.4),(0.22199999999999998,0.10245695,0.4),(0.22699999999999998,1.4000000000000012e-2,0.4)],[(0.22699999999999998,1.4000000000000012e-2,0.4),(0.23199999999999998,-7.445695e-2,0.4),(0.18745694999999998,-0.123,0.4),(6.8e-2,-0.153,0.4)],[(6.8e-2,-0.153,0.4),(-5.145695e-2,-0.183,0.4),(-0.14600000000000002,-0.17745695,0.4),(-0.195,-6.199999999999999e-2,0.4)],[(-0.195,-6.199999999999999e-2,0.4),(-0.24400000000000002,5.345695e-2,0.4),(-0.17945695,0.138,0.4),(-3.6e-2,0.174,0.4)]]
# cl4 = [[(7.999999999999998e-3,0.12300000000000001,1),(0.12245695,0.13,1),(0.19399999999999998,8.045695000000001e-2,1),(0.20299999999999999,-2.999999999999989e-3,1)],[(0.20299999999999999,-2.999999999999989e-3,1),(0.212,-8.645694999999999e-2,1),(0.17945694999999998,-0.11199999999999999,1),(3.6000000000000004e-2,-0.11699999999999999,1)],[(3.6000000000000004e-2,-0.11699999999999999,1),(-0.10745695,-0.122,1),(-0.157,-0.13445695,1),(-0.17,-2.7999999999999997e-2,1)],[(-0.17,-2.7999999999999997e-2,1),(-0.18300000000000002,7.845695000000001e-2,1),(-0.10645695,0.116,1),(7.999999999999998e-3,0.12300000000000001,1)]]
#
# controlCurves = [cl1, cl2, cl3, cl4]
