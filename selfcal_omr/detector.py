import math
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple

from selfcal_omr.helper import Point
import selfcal_omr.game as game


# import your internal modules properly
from selfcal_omr.feature import Feature, remove_junk_isoforest
from selfcal_omr.ContourTools import (
    Solidity, AspectRatio, Closeness,
    Centroid, Circularity,
    contour_depth, Findcontours,
    Estimate_grid_dx, Estimate_grid_dy, kmeans_1d
)
from selfcal_omr.rowgroup import (
    Cluster_to_Columns, Cluster_to_Rows,
    MakeBlocks, FindBlockMissingPoints
)
from dataclasses import dataclass


@dataclass
class PreprocessedImage:
    pil: object
    dpi: tuple
    img_bgr: np.ndarray
    gray: np.ndarray
    blur: np.ndarray
    thresh: np.ndarray
    contours: list
    hierarchies: np.ndarray
    depths: list
    h:int
    w:int   
    
@dataclass
class OMRConfig:
    min_bubble_cm: Tuple[float, float] = (0.25, 0.5)
    max_bubble_cm: Tuple[float, float] = (0.5, 1.0)


class OMRDetector:
    def __init__(self, config: OMRConfig = OMRConfig()):
        self.config = config

    def _cm_to_pixel(self, cm, dpi):
        dpi_x, dpi_y = dpi
        return (
            int(round(cm[0] / 2.54 * dpi_x)),
            int(round(cm[1] / 2.54 * dpi_y))
        )
        
    def isolation_detect(self, image_path: str):
        pre = self._preprocess(image_path)
        contours = pre.contours
        hierarchies = pre.hierarchies
        good, junk, _,_ = remove_junk_isoforest(
            [Feature(i, [Solidity(cnt), AspectRatio(cnt), 
                         #round(cv2.arcLength(cnt, True)/(w*h), int(math.log10(w*h))),  
                        Closeness(cnt),  
                        Centroid(cnt), Circularity(cnt)])
             for i,cnt in enumerate(contours)]
            , contamination=0.5)
        good,centers = kmeans_1d(good,3,  key=lambda g: cv2.contourArea(contours[g.id]))
        gathering_points=[]
        contours_size= np.empty((0,2))
        hulls=()
        for feature in good[1]:
            cnt = contours[feature.id]
            x_c,y_c,w_c,h_c = cv2.boundingRect(cnt)
            placed = False
            for ointX,ointY in gathering_points:
                if(math.fabs(ointX-x_c-w_c/2)<5 and math.fabs(ointY-y_c-h_c/2)<5):
                    placed = True
                    break
            if(not placed):
                contours_size = np.vstack([contours_size, [w_c,h_c]])
                hulls = hulls + (cv2.convexHull(cnt),)
                gathering_points.append(Point(x_c + w_c/2, y_c + h_c/2))   
                

        grid_dx, grid_dy = 1.5 * np.mean(contours_size, axis=0)
        teta, success = game.do([p.to_numpy() for p in gathering_points], 50, 2*max(grid_dx,grid_dy))
        rotated_points= game.Rotate(gathering_points, np.array([pre.w/2,pre.h/2]), 90 - abs(teta))

        #4.Finding columns along the new estimation of vertical grid space
        columns= Cluster_to_Columns(rotated_points, 12)
        columns = [column for column in columns if len(column) > 5]
        grid_dy =  Estimate_grid_dy(columns)

        #5.Finding rows and the new estimation of horizontial grid space
        rows= Cluster_to_Rows(rotated_points, 10)
        rows = [row for row in rows if len(row) > 1]
        grid_dx = Estimate_grid_dx(rows)


        #6.Find Blocks of bubbles in answershhet
        blocks = MakeBlocks(rows, grid_dx, grid_dy)

        #7.Finding and showing Missing Bubbles
        missing = []
        for block in blocks:
            missing.extend(
                FindBlockMissingPoints(block, grid_dx, grid_dy)
            )
        if missing:
            missing__= game.Rotate(missing, np.array([pre.w/2,pre.h/2]), -90 + abs(teta))
        else:
            missing__=[]

        return (
            pre.img_bgr,
            blocks,
            missing__,
            hulls,
            gathering_points,
            float(grid_dx),
            float(grid_dy),
            teta,
            int( 100 *success / 50  )
        )
        
        
    def normal_detect(self, image_path: str):
        pre = self._preprocess(image_path)
        min_bubble_size = self._cm_to_pixel(
            self.config.min_bubble_cm,
            pre.dpi
            )
        contours = pre.contours
        hierarchies = pre.hierarchies

        gathering_points, hulls, contours_size = Findcontours(
            contours, hierarchies, min_bubble_size
        )

        grid_dx, grid_dy = 1.5 * np.mean(contours_size, axis=0)
        teta, success = game.do([p.to_numpy() for p in gathering_points], 50, 2*max(grid_dx,grid_dy))
        rotated_points= game.Rotate(gathering_points, np.array([pre.w/2,pre.h/2]), 90 - abs(teta))

        #4.Finding columns along the new estimation of vertical grid space
        columns= Cluster_to_Columns(rotated_points, 12)
        columns = [column for column in columns if len(column) > 5]
        grid_dy =  Estimate_grid_dy(columns)

        #5.Finding rows and the new estimation of horizontial grid space
        rows= Cluster_to_Rows(rotated_points, 10)
        rows = [row for row in rows if len(row) > 1]
        grid_dx = Estimate_grid_dx(rows)


        #6.Find Blocks of bubbles in answershhet
        blocks = MakeBlocks(rows, grid_dx, grid_dy)

        #7.Finding and showing Missing Bubbles
        missing = []
        for block in blocks:
            missing.extend(
                FindBlockMissingPoints(block, grid_dx, grid_dy)
            )
        if missing:
            missing__= game.Rotate(missing, np.array([pre.w/2,pre.h/2]), -90 + abs(teta))
        else:
            missing__=[]

        return (
            pre.img_bgr,
            blocks,
            missing__,
            hulls,
            gathering_points,
            float(grid_dx),
            float(grid_dy),
            teta,
            int( 100 *success / 50  )
        )

    def _preprocess(self, image_path: str):
        # Load image
        pil = Image.open(image_path)
        dpi = pil.info.get("dpi", (72, 72))

        # Convert to OpenCV format
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Threshold
        _, thresh = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Contours
        contours, hierarchies = cv2.findContours(
            thresh, cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchies is not None:
            hierarchies = hierarchies[0]

        # Depths
        depths = []
        if hierarchies is not None:
            depths = [
                contour_depth(hierarchies, i)
                for i in range(len(contours))
            ]

        # Return structured data
        return PreprocessedImage(
            pil=pil,
            dpi=dpi,
            img_bgr=img_bgr,
            gray=gray,
            blur=blur,
            thresh=thresh,
            contours=contours,
            hierarchies=hierarchies,
            depths=depths,
            h=h,
            w=w
        )

