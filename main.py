"""
Created on 17.12.2021 by Andrea Gebek.
"""

from PIL import Image
import PIL
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from urllib.request import urlretrieve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import axes_grid1
import datetime
from selenium import webdriver
from bs4 import BeautifulSoup
import os
import scipy.interpolate
import math
import multiprocessing as mp
from functools import partial

os.environ['MOZ_HEADLESS'] = '1' # Prevent the opening of firefox for the web scraping
PIL.Image.MAX_IMAGE_PIXELS = 1e11 # Allow the loading of the maps which contain a lot of pixels

startTime = datetime.datetime.now()

# Plotting aesthetics

matplotlib.rcParams['axes.linewidth'] = 2.5
matplotlib.rcParams['xtick.major.size'] = 10
matplotlib.rcParams['xtick.minor.size'] = 6
matplotlib.rcParams['xtick.major.width'] = 2.5
matplotlib.rcParams['xtick.minor.width'] = 1.5
matplotlib.rcParams['ytick.major.size'] = 10
matplotlib.rcParams['ytick.minor.size'] = 6
matplotlib.rcParams['ytick.major.width'] = 2.5
matplotlib.rcParams['ytick.minor.width'] = 1.5
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams.update({'font.size': 26, 'font.weight': 'bold'})



def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs): # Add colorbar to a plot which matches the graph size
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


# Essential functions

def makeSoup(url, driver):

    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content, features = 'lxml')

    return soup

def scrapeGPS(x_center, y_center):

    urlGPS = 'https://tools.retorte.ch/map/?swissgrid=' + str(int(x_center)) + ',' + str(int(y_center)) + '&zoom=15'

    driver = webdriver.Firefox()

    GPS_soup = makeSoup(urlGPS, driver)
    GPS_html = GPS_soup.find('td', {'id': 'wgs84_output'})

    latitude_center = float(GPS_html.text[:-1].split()[0])
    longitude_center = float(GPS_html.text[:-1].split()[2])

    driver.quit()

    return latitude_center, longitude_center

def scrapeMoonCoords(startHour, endHour, startMin, endMin, timeResolution, startDate, latitude_center, longitude_center):

    if startHour > endHour:

        endHour += 24

    minArray= np.arange(startHour * 60. + startMin, endHour * 60. + endMin, timeResolution)

    minList = [int(x) for x in minArray % 60.]
    hourList = [int(x) for x in minArray / 60.]

    elevationList = []
    azimuthList = []
    timeList = []

    driver = webdriver.Firefox()

    for idx, hour in enumerate(hourList):

        url_mooncalc = 'https://www.mooncalc.org/#/' + str(latitude_center) + ',' + str(longitude_center) + ',2/' + startDate + '/' + str(hour) + ':' + str(minList[idx]) + '/0/0'

        mooncalc_soup = makeSoup(url_mooncalc, driver)

        elevation_html = mooncalc_soup.find('span', attrs = {'id': 'sunhoehe'})
        azimuth_html = mooncalc_soup.find('span', attrs = {'id': 'azimuth'})

        elevationList.append(float(elevation_html.text[:-1]))
        azimuthList.append(float(azimuth_html.text[:-1]))

        if hour >= 24:

            hour -= 24

        if minList[idx] < 10:

            timeList.append(str(hour) + ':0' + str(minList[idx]))

        else:

            timeList.append(str(hour) + ':' + str(minList[idx]))   
    
    driver.quit()

    return elevationList, azimuthList, timeList

def get3Ddata(x_center, y_center, deltaRaytracing_x, deltaRaytracing_y, raytracingResolution):

    xBorders = (float(x_center) / 1000. - float(deltaRaytracing_x) / 2., float(x_center) / 1000. + float(deltaRaytracing_x) / 2.)
    yBorders = (float(y_center) / 1000. - float(deltaRaytracing_y) / 2., float(y_center) / 1000. + float(deltaRaytracing_y) / 2.)

    x_km_array = np.arange(math.floor(xBorders[0]), math.ceil(xBorders[1]), 1)
    y_km_array = np.arange(math.floor(yBorders[0]), math.ceil(yBorders[1]), 1)

    r_3Dlist = []

    for x in x_km_array:

        for y in y_km_array:

            link3d = 'https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2019_' + str(x) + '-' + str(y) + '/swissalti3d_2019_' + str(x) + '-' + str(y) + '_2_2056_5728.xyz.zip'

            resp = urlopen(link3d)
            zipfile = ZipFile(BytesIO(resp.read()))
            filename = zipfile.namelist()[0]

            for line in zipfile.open(filename).readlines()[1:]:

                coords = [float(i) for i in line.decode('utf-8').split()]

                r_3Dlist.append(coords)

    r_3D = np.array(r_3Dlist)

    # Degrade the resolution

    coarseGrid_x = np.arange(np.min(r_3D[:, 0]) / 1000., np.max(r_3D[:, 0]) / 1000., raytracingResolution)
    coarseGrid_y = np.arange(np.min(r_3D[:, 1]) / 1000., np.max(r_3D[:, 1]) / 1000., raytracingResolution)

    coarseGrid_X, coarseGrid_Y = np.meshgrid(coarseGrid_x, coarseGrid_y, indexing = 'ij')

    coarseGrid = np.vstack((coarseGrid_X.flatten(), coarseGrid_Y.flatten())).T

    altitudes = scipy.interpolate.griddata(r_3D[:, 0:2] / 1000., r_3D[:, 2] / 1000., coarseGrid)

    return coarseGrid, altitudes

def getMaps(x_center, y_center, deltaHike_x, deltaHike_y):

    mapyears = np.arange(1984, 2023, 1)[::-1]

    xBorders = (float(x_center) / 1000. - float(deltaHike_x) / 2., float(x_center) / 1000. + float(deltaHike_x) / 2.)
    yBorders = (float(y_center) / 1000. - float(deltaHike_y) / 2., float(y_center) / 1000. + float(deltaHike_y) / 2.)

    xMapBorders_indices = (math.floor((xBorders[0] - 2480.) / 17.5), math.ceil((xBorders[1] - 2480.) / 17.5))
    yMapBorders_indices = (math.floor((1302. - yBorders[1]) / 12.), math.ceil((1302. - yBorders[0]) / 12.))

    xMap_indices = np.arange(xMapBorders_indices[0], xMapBorders_indices[1], 1)
    yMap_indices = np.arange(yMapBorders_indices[0], yMapBorders_indices[1], 1)

    xMap = np.arange(2480 + 17.5 * xMapBorders_indices[0], 2480 + 17.5 * xMapBorders_indices[1], 0.00125)
    yMap = np.arange(1302 - 12. * yMapBorders_indices[1], 1302 - 12. * yMapBorders_indices[0], 0.00125)

    x_argmin = np.max(np.argwhere(xMap < xBorders[0]))
    x_argmax = np.min(np.argwhere(xMap > xBorders[1])) - 1
    y_argmin = np.max(np.argwhere(yMap < yBorders[0])) + 1
    y_argmax = np.min(np.argwhere(yMap > yBorders[1]))

    mapImageTotal = np.zeros((len(yMap), len(xMap), 3), dtype = int)

    for idx_x, x in enumerate(xMap_indices):

        for idx_y, y in enumerate(yMap_indices):

            mapIdx = 1000 + y * 20 + x

            for year in mapyears:

                try:

                    linkmap = 'https://data.geo.admin.ch/ch.swisstopo.pixelkarte-farbe-pk25.noscale/swiss-map-raster25_' + str(year) + '_' + str(mapIdx) + '/swiss-map-raster25_' + str(year) + '_' + str(mapIdx) + '_krel_1.25_2056.tif'
                    resp = urlretrieve(linkmap)[0]
                    im = Image.open(resp)
                    imarr = np.array(im)[::-1, :, :]

                    mapImageTotal[((len(yMap_indices) - idx_y - 1) * 9600) : ((len(yMap_indices) - idx_y) * 9600), (idx_x * 14000) : ((idx_x + 1) * 14000), :] = imarr

                    break

                except:
                    
                    continue

    mapImage = mapImageTotal[y_argmin : (y_argmax + 1), x_argmin : (x_argmax + 1), :]
    mapExtent = [xMap[x_argmin], xMap[x_argmax], yMap[y_argmin], yMap[y_argmax]]

    mapGrid_x = xMap[x_argmin : (x_argmax + 1)]
    mapGrid_y = yMap[y_argmin : (y_argmax + 1)]

    mapGrid_X, mapGrid_Y = np.meshgrid(mapGrid_x, mapGrid_y, indexing = 'ij')

    mapGrid = np.vstack((mapGrid_X.flatten(), mapGrid_Y.flatten())).T

    return mapImage, mapExtent, mapGrid

def traceRays(moonCoords, args):

    mapImage = args['mapImage']
    mapGrid = args['mapGrid']
    coarseGrid = args['coarseGrid']
    altitudes = args['altitudes']
    x_center = args['x_center']
    y_center = args['y_center']
    deltaHike_x = args['deltaHike_x']
    deltaHike_y = args['deltaHike_y']
    raytracingResolution = args['raytracingResolution']

    res = raytracingResolution / 2.

    xBorders = (float(x_center) / 1000. - float(deltaHike_x) / 2., float(x_center) / 1000. + float(deltaHike_x) / 2.)
    yBorders = (float(y_center) / 1000. - float(deltaHike_y) / 2., float(y_center) / 1000. + float(deltaHike_y) / 2.)

    azimuth_rad = moonCoords[0] * np.pi / 180.
    elevation_rad = moonCoords[1] * np.pi / 180.

    if elevation_rad < 0:

        block = np.ones_like(mapImage[:, :, 0]) * 100

    
    else:

        SELHIKE = (coarseGrid[:, 0] >= xBorders[0]) * (coarseGrid[:, 0] <= xBorders[1]) * (coarseGrid[:, 1] >= yBorders[0]) * (coarseGrid[:, 1] <= yBorders[1])

        block = []

        for idx in range(np.size(coarseGrid[SELHIKE], axis = 0)):

            delta_x = coarseGrid[:, 0] - coarseGrid[:, 0][SELHIKE][idx]
            delta_y = coarseGrid[:, 1] - coarseGrid[:, 1][SELHIKE][idx]

            SEL0 = (np.arctan((delta_x - res) / (delta_y - res)) < azimuth_rad) * (np.arctan((delta_x + res) / (delta_y - res)) > azimuth_rad) * (delta_x == 0) * (delta_y > 0) 
            # Cells directly above pos_init. Note that the first condition here is always fulfilled if the last two conditions are true, I have to write it without 
            # the 2*pi term because of the dicontinuity in the azimuth at 360 degrees.
            SEL1 = (np.arctan((delta_x - res) / (delta_y + res)) < azimuth_rad) * (np.arctan((delta_x + res) / (delta_y - res)) > azimuth_rad) * (delta_x > 0) * (delta_y > 0)
            SEL2 = (np.arctan((delta_x - res) / (delta_y + res)) < azimuth_rad) * (np.arctan((delta_x - res) / (delta_y - res)) + np.pi > azimuth_rad) * (delta_x > 0) * (delta_y == 0)
            SEL3 = (np.arctan((delta_x + res) / (delta_y + res)) + np.pi < azimuth_rad) * (np.arctan((delta_x - res) / (delta_y - res)) + np.pi > azimuth_rad) * (delta_x > 0) * (delta_y < 0)
            SEL4 = (np.arctan((delta_x + res) / (delta_y + res)) + np.pi < azimuth_rad) * (np.arctan((delta_x - res) / (delta_y + res)) + np.pi > azimuth_rad) * (delta_x == 0) * (delta_y < 0)
            SEL5 = (np.arctan((delta_x + res) / (delta_y - res))+ np.pi < azimuth_rad) * (np.arctan((delta_x - res) / (delta_y + res)) + np.pi > azimuth_rad) * (delta_x < 0) * (delta_y < 0)
            SEL6 = (np.arctan((delta_x + res) / (delta_y - res)) + np.pi < azimuth_rad) * (np.arctan((delta_x + res) / (delta_y + res)) + 2 * np.pi > azimuth_rad) * (delta_x < 0) * (delta_y == 0)
            SEL7 = (np.arctan((delta_x - res) / (delta_y - res)) + 2 * np.pi < azimuth_rad) * (np.arctan((delta_x + res) / (delta_y + res)) + 2 * np.pi > azimuth_rad) * (delta_x < 0) * (delta_y > 0)
            
            SELxy = SEL0 + SEL1 + SEL2 + SEL3 + SEL4 + SEL5 + SEL6 + SEL7
        

            SEL = (altitudes[SELxy] > altitudes[SELHIKE][idx] + np.tan(elevation_rad) * np.sqrt(delta_x[SELxy]**2 + delta_y[SELxy]**2))

            if len(altitudes[SELxy][SEL]) > 0:

                block.append(100)

            else:

                block.append(255)

        block = scipy.interpolate.griddata(coarseGrid[SELHIKE], block, mapGrid, method = 'nearest')

        block = np.reshape(block, (np.size(mapImage, axis = 1), np.size(mapImage, axis = 0))).T[:, :]

    return block


def createMovie(blockList, mapImage, mapGrid, mapExtent, timeList, pathFigure):

    fig = plt.figure(figsize = (30, 20))
    ax = fig.add_subplot(111)

    ims = []

    for idx, block in enumerate(blockList):

        FINAL = np.concatenate((mapImage, block[:, :, None]), axis = 2)

        im = ax.imshow(FINAL, extent = mapExtent)

        ax.set_xlabel(r'$x\,[\mathrm{km}]$')
        ax.set_ylabel(r'$y\,[\mathrm{km}]$')

        title = plt.text(0.5, 1.01, timeList[idx], horizontalalignment = 'center', verticalalignment = 'bottom', transform = ax.transAxes)

        ax.minorticks_on()
        ax.tick_params(which = 'both', direction = 'in', right = True, top = True)

        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 1000)
    writervideo = animation.FFMpegWriter(fps=1, bitrate = 5000)
    ani.save(pathFigure, writer = writervideo)

    return 0


"""
Code execution
"""
if __name__ == '__main__':

    # Input parameters

    x_center = 2631500 # x-coordinate of the center of the hiking region in the Swiss coordinate system
    y_center = 1176000 # y-coordinate of the center of the hiking region in the Swiss coordinate system

    deltaHike_x = 3. # Extent of the hiking region in x-direction (in km)
    deltaHike_y = 2. # Extent of the hiking region in y-direction (in km)

    deltaRaytracing_x = 10. # Extent of the region taken into account for the raytracing in x-direction (in km)
    deltaRaytracing_y = 10. # Extent of the region taken into account for the raytracing in y-direction (in km)
    raytracingResolution = 0.025 # Resolution of the raytracing in km

    startDate = '2022.06.11' # yyyy.mm.dd
    startHour = 21
    startMin = 15

    endHour = 25
    endMin = 30

    timeResolution = 10 # In minutes

    pathFigure = '/Users/agebek/Downloads/mondHabkern.mp4'



    # Web scraping

    latitude_center, longitude_center = scrapeGPS(x_center, y_center)

    elevationList, azimuthList, timeList = scrapeMoonCoords(startHour, endHour, startMin, endMin, timeResolution, startDate, latitude_center, longitude_center)

    print('Time elapsed for Web scraping:', datetime.datetime.now() - startTime)


    # Get the 3D-data and maps for the selected location from swisstopo


    mapImage, mapExtent, mapGrid = getMaps(x_center, y_center, deltaHike_x, deltaHike_y)

    print('Time elapsed for map download:', datetime.datetime.now() - startTime)



    coarseGrid, altitudes = get3Ddata(x_center, y_center, deltaRaytracing_x, deltaRaytracing_y, raytracingResolution)

    print('Time elapsed for 3D data download:', datetime.datetime.now() - startTime)



    moonCoords = np.column_stack((azimuthList, elevationList))


    args = {'mapImage': mapImage, 'mapGrid': mapGrid, 'coarseGrid': coarseGrid, 'altitudes': altitudes, 'x_center': x_center, 'y_center': y_center, 'deltaHike_x': deltaHike_x,
    'deltaHike_y': deltaHike_y, 'raytracingResolution': raytracingResolution}

    blockList = []

    with mp.Pool(processes = 12) as pool:

        blockList = pool.map(partial(traceRays, args = args), moonCoords)

    pool.close()
    pool.join()

    print('Time elapsed for raytracing:', datetime.datetime.now() - startTime)

    createMovie(blockList, mapImage, mapGrid, mapExtent, timeList, pathFigure)

    print('Execution time for the script:', datetime.datetime.now() - startTime)