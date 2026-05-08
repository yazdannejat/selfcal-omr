import math
from operator import attrgetter
import statistics
import cv2
import numpy as np

from selfcal_omr.feature import Point4D
from selfcal_omr.helper import Point

template1 = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, .5, -1, -1, -1, -1, -1, -1, -1, -1, .5, 1],
       [1, .5, -1, -1, -1, -1, -1, -1, -1, -1, .5, 1],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
            
template2 = np.array([[0, .5, 1, 1, 1, 1, 1, 1, 1, 1, .5, 0],
       [.5, 1, .5, -1, -1, -1, -1, -1, -1, .5, 1, .5],
       [1, .5, -1, -1, -1, -1, -1, -1, -1, -1, .5, 1],
       [1, .5, -1, -1, -1, -1, -1, -1, -1, -1, .5, 1],
       [.5, 1, .5, -1, -1, -1, -1, -1, -1, .5, 1, .5],
       [0, .5, 1, 1, 1, 1, 1, 1, 1, 1, .5, 0]])

template1 = np.array([[.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, .5],
       [1, .5, 0, 0, 0, 0, 0, 0, 0, 0, .5, 1],
       [1, .5, 0, 0, 0, 0, 0, 0, 0, 0, .5, 1],
       [.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, .5]])
            
template2 = np.array([[.5, .5, 1, 1, 1, 1, 1, 1, 1, 1, .5, .5],
       [.5, 1, .5, 0, 0, 0, 0, 0, 0, .5, 1, .5],
       [1, .5, 0, 0, 0, 0, 0, 0, 0, 0, .5, 1],
       [1, .5, 0, 0, 0, 0, 0, 0, 0, 0, .5, 1],
       [.5, 1, .5, 0, 0, 0, 0, 0, 0, .5, 1, .5],
       [.5, .5, 1, 1, 1, 1, 1, 1, 1, 1, .5, .5]])

template3 = np.array([[1., 1., 1., 1., 1., 1., 1.],
       [1., 0.5, 0.5, 0.5, 0.5, 0.5, 1.],
       [1., 0.5, 0., 0., 0., 0.5, 1.],
       [1., 0.5, 0., 0., 0., 0.5,  1.],
       [1., 0.5, 0.5, 0.5, 0.5,  0.5, 1.],
       [1., 0.5, 0.5, 0.5, 0.5,  0.5, 1.],
       [1., 1., 1., 1., 1., 1., 1.]])
template = (template1, template2, template3)

def contour_depth(hierarchy, idx):
    depth = 0
    parent = hierarchy[idx][3]
    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]
    return depth

def compute_depth(hierarchy, idx):
    depth = 0
    parent = hierarchy[idx][3]
    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]
    return depth
    
def GetDepths(hierarchy, contours):
    return[
        compute_depth(hierarchy, i)
        for i in range(len(contours))]
    
def GetDeptDict(hierarchy, contours, depth):
    contour_depths = GetDepths(hierarchy, contours)
    classified = {}
    for i, depth in enumerate(contour_depths):
        classified.setdefault(depth, []).append(i)
    return classified

def showContours(img, hierarchy, contours):
    colors = [
    (0,0,255),   # red for level 0
    (0,255,0),   # green for level 1
    (255,0,0),   # blue for level 2
    (0,255,255), # cyan for level 3
    ]
    contour_depths = GetDepths(hierarchy, contours)
    for i, cnt in enumerate(contours):
        depth = contour_depths[i]
        col = colors[depth % len(colors)]
        cv2.drawContours(img, contours, i, col, 2)
        
def group_by_closeness(collection, key, threshold):
    groups = []
    for item in collection:
        value = key(item)
        placed = False

        for group in groups:
            # Compare with any representative of the group
            rep = group[0]
            if abs(key(rep) - value) <= threshold:
                group.append(item)
                placed = True
                break

        if not placed:
            groups.append([item])  # create a new group

    return groups

def group_by_categories(collection, key, categories):
    groups = {cat: [] for cat in categories}
    for item in collection:
        value = key(item)
        placed = False

        for cat in categories:
            # Compare with any representative of the group            
            if value <= cat:
                groups[cat].append(item)
                Placed = True
                break
        if not Placed :
            groups[-1].append(item)        

    return groups

def kmeans_1d(data, k, key=lambda z: z, iterations=5):
    # Extract numeric values
    values = np.array([key(x) for x in data], dtype=float)

    # Sort by value but keep original items
    idx = np.argsort(values)
    values = values[idx]
    data = [data[i] for i in idx]

    # Initialize centers from evenly spaced samples
    centers = values[np.linspace(0, len(values)-1, k, dtype=int)]

    for _ in range(iterations):
        # Assign each item to nearest center
        clusters = [[] for _ in range(k)]
        cluster_vals = [[] for _ in range(k)]

        for item, val in zip(data, values):
            ci = np.argmin(np.abs(centers - val))
            clusters[ci].append(item)
            cluster_vals[ci].append(val)

        # Recompute centers
        for i in range(k):
            if cluster_vals[i]:  # avoid empty clusters
                centers[i] = np.mean(cluster_vals[i])

    return clusters, centers

def kmeans__1d(x, k, key=lambda z: z):
    x = np.sort(x)
    centers = x[np.linspace(0, len(x)-1, k, dtype=int)]
    for _ in range(5):  # few iterations are enough in 1D
        clusters = [[] for _ in range(k)]
        for v in x:
            idx = np.argmin(abs(centers - v))
            clusters[idx].append(v)
        centers = np.array([np.mean(c) for c in clusters])
    return clusters, centers

def dbscan_1d(x, eps):
    x = sorted(x)
    clusters = []
    cluster = [x[0]]

    for i in range(1, len(x)):
        if abs(x[i] - x[i-1]) <= eps:
            cluster.append(x[i])
        else:
            clusters.append(cluster)
            cluster = [x[i]]
    clusters.append(cluster)
    return clusters

def cluster_by_relation(collection, relation):
    clusters = []
    for item in collection:
        placed = False
        for group in clusters:
            # If item is close to ANY member of group
            if any(relation(item, member) for member in group):
                group.append(item)
                placed = True
                break

        if not placed:
            clusters.append([item])
    return clusters

def abundant_feature(collection, key=lambda x: x):
    freq = {}
    for item in collection:
        k = key(item)
        freq[k] = freq.get(k, 0) + 1
    if not freq:
        return None
    dominant_feature = max(freq, key=freq.get)
    for item in collection:
        if key(item) == dominant_feature:
            return item
    return None

def get_rect_cnts(contours):
    rect_cnts = []
    for cnt in contours:
        # approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        # if the approximated contour is a rectangle ...
        if len(approx) == 4:
            # append it to our list
            rect_cnts.append(approx)
    # sort the contours from biggest to smallest
    rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)    
    return rect_cnts



def ClusterPoints(points, k):
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    hulls=[]
    for j in range(k):
        values = np.asarray([points[i] for i in range(len(labels)) if labels[i]==j])
        values = values.reshape(-1,1,2)
        print("{}: {}".format(j, values.shape))
        x,y,w,h =cv2.boundingRect(cv2.convexHull(values))
        hulls.append([(x,y),(x+w, y+h)])
    return hulls

def Countor_Is_Rectangle(contour):
    # approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.1* peri, True)
    # if the approximated contour is a rectangle ...
    if len(approx) <= 8:
        return True    
    return False

def Estimate_grid_dx(rows):
    dy=[]    
    dx=[]
    for row in rows:
        _row = sorted(row, key= attrgetter("x") )
        for i in range(1, len(_row)):
            dy.append(math.fabs(_row[i].x - _row[i-1].x))
        dx.append(statistics.median(dy))
        dy=[]
    return statistics.median(dx)
    
def Estimate_grid_dy(columns):
    dy=[]   
    dx=[]
    for column in columns:
        _column = sorted(column, key= attrgetter("y") )
        for i in range(1, len(_column)):
            dx.append(math.fabs(_column[i].y - _column[i-1].y))
        if(len(_column) > 1):
            dy.append(statistics.median(dx))
        dx=[]
    return statistics.median(dy)

def Dist(contour1, contour2):
    sumx_1,sumy_1 = Momentum(contour1)
    sumx_2,sumy_2 = Momentum(contour2)
    return math.sqrt(math.pow(sumx_1 - sumx_2, 2) + math.pow(sumy_1 - sumy_2, 2))

def Contour_Likelihood(contour,i):
    mat = Contour_To_Matrice(contour,i)
    temp = mat * template[i]
    sum = 0
    for x in temp.flatten(): sum+=x
    return sum

def Contour_To_Matrice(contour,i):
    mat= np.zeros(template[i].shape, dtype=int)
    sizeX, sizeY = template[i].shape
    x,y,w,h = cv2.boundingRect(contour)
    for point in contour:
        mat[int(math.floor((point[0][0]-x)*sizeX/w)), int(math.floor((point[0][1]-y)*sizeY/h))] = 1
    return mat
def Closure_Matrice(contours,distance):
    mat = np.zeros(len(contours),len(contours))
    
def Solidity(cnt):
    hullarea = cv2.contourArea(cv2.convexHull(cnt))
    area = cv2.contourArea(cnt)
    if(hullarea == 0): solidity = -1
    else: solidity = area / hullarea  
    return solidity 

def Closeness(cnt):
    if cnt is None or len(cnt) == 0:
        return -1
    cnt = np.array(cnt, dtype=np.float32)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    #box = np.int0(box)
    box = box.reshape((-1,1,2)).astype(np.int32)
    # Compute how well contour matches that rectangle
    # measure average distance between contour and fitted box
    dist_sum = 0
    for p in cnt:
         x = float(p[0][0])
         y = float(p[0][1])
         d = cv2.pointPolygonTest(box, (x, y), True)
         dist_sum += abs(d)
    return dist_sum / len(cnt)

def AspectRatio(cnt):
    _,_,w,h = cv2.boundingRect(cnt) 
    if h==0: return -1  
    return w/float(h)

def Circularity(cnt):
    p = cv2.arcLength(cnt, True)
    s = cv2.contourArea(cnt)
    if p==0: return -1
    return (4 * math.pi * s) / (p * p)

def _Centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

def Centroid(contour):
    cx,cy = Momentum(contour)    
    x, y, w, h = cv2.boundingRect(contour)
    rx = x + w / 2
    ry = y + h / 2
    dist = np.sqrt((cx - rx)**2 + (cy - ry)**2)
    diag = np.sqrt(w*w + h*h)
    return dist / diag

def Momentum(contour):
    sumx=sumy = 0
    for point in contour:
        sumx+=point[0][0]
        sumy+=point[0][1]
    sumx=sumx/len(contour)
    sumy=sumy/len(contour)
    return [sumx, sumy]

def Normalized_Momentum(contour):
    x, y, w, h = cv2.boundingRect(contour)
    sumx=sumy = 0
    for point in contour:
        sumx+=(point[0][0]-x)/w
        sumy+=(point[0][1]-y)/h
    sumx=sumx/len(contour)
    sumy=sumy/len(contour)
    return np.sqrt((.5 - sumx)**2 + (.5 - sumy)**2)

def CompareKey(cnt):
    m = Momentum(cnt)
    x_c,y_c,w_c,h_c = cv2.boundingRect(cnt)
    p1 = Point(x_c + float(w_c)/2, y_c + float(h_c)/2)
    p2 =Point(m[0], m[1])
    return p1.distance_to(p2)

def get_sorted_Contour(contours):
    return sorted(contours, key=CompareKey)

def feature1(i,cnt):
    return Point4D(i, CompareKey(cnt), 0 ,0 ,0)
def feature1_1(i, cnt):
    return Point4D(i, cv2.arcLength(cnt, True), Contour_Likelihood(cv2.convexHull(cnt),1) ,0 ,0)
def feature2(i, cnt):
    return Point4D(i, CompareKey(cnt), cv2.arcLength(cnt, True) ,0 ,0)
def feature3(i, cnt):
    return Point4D(i, CompareKey(cnt), cv2.contourArea(cnt) ,cv2.arcLength(cnt, True) ,0)
def feature4(i, cnt):
    return Point4D(i, CompareKey(cnt), cv2.contourArea(cnt) ,cv2.arcLength(cnt, True) , Solidity(cnt))
def feature(cnt, i =0):
    return Point4D(i, Centroid(cnt), Solidity(cnt) ,Circularity(cnt) , AspectRatio(cnt))


def cm_to_pixel(cm, dpi):
    dpi_x, dpi_y = dpi
    return (
        int(round(cm[0] / 2.54 * dpi_x)),
        int(round(cm[1] / 2.54 * dpi_y))
    )

def Findcontours(contours, hierarchies, min_bubble_size):
    gathering_points=[]
    hulls= ()
    contours_size= np.empty((0,2))
    for i, contour_ in enumerate(contours):
        if hierarchies[i][3] < 0 and hierarchies[i][2] >= 0:
            continue
        p = cv2.arcLength(contour_, True)

        #if True:
        if(p > sum(min_bubble_size)
        and 1.4 <= AspectRatio(contour_) <= 2.5  
        and Solidity(contour_) > .9  
        and Centroid(contour_) < .1 
        and .65 <= Circularity(contour_) <= .85
        #and depths[i] == 1
        ):
            x_c,y_c,w_c,h_c = cv2.boundingRect(contour_)
            
            placed = False
            hull = cv2.convexHull(contour_)
        
            for ointX,ointY in gathering_points:
                if(math.fabs(ointX-x_c-w_c/2)<5 and math.fabs(ointY-y_c-h_c/2)<5):
                    placed = True
                    break
            if(not placed):
                contours_size = np.vstack([contours_size, [w_c,h_c]])
                hulls = hulls + (hull,)
                gathering_points.append(Point(x_c + w_c/2, y_c + h_c/2))   
    return gathering_points, hulls, contours_size