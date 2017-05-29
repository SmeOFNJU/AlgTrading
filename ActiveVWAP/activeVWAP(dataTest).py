# -*- coding: utf-8 -*-
import copy
import pandas
import numpy as np
import pywt
import math
import matplotlib.pyplot as pl

from gmsdk.api import StrategyBase
from gmsdk import md
from gmsdk import td

Market = 'SHSE'
StockId = '600363'
TradeDate = '2016-12-28'
Subscribe_symbol = Market + '.' + StockId + '.bar.60'
StartTime = TradeDate + ' ' + '09:30:00'
EndTime = TradeDate + ' ' + '15:30:00'
StartHistoryDay = '2016-01-01'
EndHistoryDay = '2016-12-27'
CurrentDay = '2016-12-28'

HistoryDays = 30 # 历史数据N天
Intervals = 48 # 一个大时间片的长度
IntervalWid = 300 #一个大时间片的宽度，即5min
IntervalWidMin = 5 #一个大时间片的宽度，即5min（以min表示）
TargetQuantity = 100000 # 今日目标股数
RemindQuantity = TargetQuantity
fiveMinQuantity = 0
TargetVolList = []

AmountMatrix = [[0 for col in range(48)] for row in range(HistoryDays)]
VolumeMatrix = [[0 for col in range(48)] for row in range(HistoryDays)]
FiveMinOpenPriceMatrix = []
TodayVRList = []
FiveMinHisAveragePriceMatrix = []
IntervalBorderList = []
TradeDayPriceList = []

CurrentOpenPriceList = []
CurrentAveragePricesList = []
ForcastOpenPirceList = []
ForcastAveragePricesList = []
HistoryAveragePricesList = []

WeightList = [1 for col in range(HistoryDays)]  # 每天的权重列表
lastLittleIntervalOrderId = 0  # 上一个小时间片的委托号

# 取整数笔
def ToIntShare(quantity):
    shares = (int)(quantity / 100)
    remainder  = quantity % 100
    if (remainder < 50):
        return shares * 100
    else:
        return (shares + 1) * 100



# 1.1 返回一个timeSpan*48的矩阵，用来存放第timeSpan天第i个时间片上的交易量（历史交易日包括endDay）
def FillVolumnMatrix(stockId, startDay, endDay, timeSpan):
    global AmountMatrix
    global VolumeMatrix
    TradeDayList = []
    calendar = md.get_calendar(Market, startDay, endDay)
    #print('len(calender)', len(calendar))
    index = len(calendar) - 1
    for i in range(0, timeSpan):
        #print('index', index)
        timeOri = calendar[index].strtime
        strIndex = timeOri.index('T')
        timeAft = timeOri[0:strIndex]
        TradeDayList.insert(0, timeAft)
        index -= 1
    for day in TradeDayList:
        dayIndex = TradeDayList.index(day)
        startTime = day + ' ' + '9:30:00'
        endTime = day + ' ' + '15:00:00'
        bars = md.get_bars(stockId, IntervalWid, startTime, endTime)
        for interval in range(len(bars)):
            VolumeMatrix[dayIndex][interval] = bars[interval].volume
            AmountMatrix[dayIndex][interval] = bars[interval].amount


def GetTradeDayPriceList(stockId, tradeDay):
    tradedayPriceList = []
    startTime = tradeDay + ' ' + '9:30:00'
    endTime = tradeDay + ' ' + '15:00:00'
    bars = md.get_bars(stockId, IntervalWid, startTime, endTime)
    for interval in range(len(bars)):
        tradedayPriceList.insert(bars[interval].open)
    return tradedayPriceList


# 1.2 返回第row天第col个时间片上交易量的比例
def FillVolumnRatioMatrix(volumnMatrix):
    timeSpan = len(volumnMatrix)
    volumnRatioMatrix = [[0 for col in range(48)] for row in range(timeSpan)] # 第row天第col个时间片上交易量的比例
    for row in range(timeSpan):
        sumOfCol = 0
        for col in volumnMatrix[row]:
            sumOfCol = sumOfCol + col
        for col in range(len(volumnMatrix[row])):
            volumnRatioMatrix[row][col] = volumnMatrix[row][col] / sumOfCol
    return volumnRatioMatrix

# 1.3 返回计算当前交易日的各个时间片的权重
def CalTodayVolumnRatioList(volumnRatioMatrix, weightList):
    todayVolumnRatioList = []
    weightSum = 0
    for weight in weightList:
        weightSum += weight
    intervalCount = len(volumnRatioMatrix[0])
    for interval in range(intervalCount):
        currentIntervalRatio = 0
        for dayIndex in range(len(weightList)):
            currentIntervalRatio += weightList[dayIndex] * volumnRatioMatrix[dayIndex][interval]
            if (dayIndex == len(weightList) - 1):
                currentIntervalRatio = currentIntervalRatio / weightSum
        todayVolumnRatioList.append(currentIntervalRatio)
    return todayVolumnRatioList

# 1.4 返回当前交易日的时间片边界列表
def GetIntervalList(tradeDay, intervalSpan=5):
    intervalBorderList = []
    amStartTime = tradeDay + ' ' + '09:30:00' + '+08:00'
    pmStartTime = tradeDay + ' ' + '13:00:00' + '+08:00'
    tempAmList = pandas.date_range(amStartTime, periods=24, freq='5min')
    tempPmList = pandas.date_range(pmStartTime, periods=24, freq='5min')
    for item in range(len(tempAmList)):
        intervalBorderList.append(tempAmList[item])
    for item in range(len(tempPmList)):
        intervalBorderList.append(tempPmList[item])
    return intervalBorderList

# 1.5 返回历史交易日的5分钟均价矩阵
def GetHistoryIntervalPrice(volMatrix, amountMatrix):
    intervalPiecesCount = 48
    priceMatrix = [[0 for col in range(intervalPiecesCount)] for row in range(HistoryDays)]
    for dayIndex in range(0, HistoryDays):
        for intervalIndex in range(0, len(priceMatrix[dayIndex])):
            volume = volMatrix[dayIndex][intervalIndex]
            if volume != 0:
                priceMatrix[dayIndex][intervalIndex] = amountMatrix[dayIndex][intervalIndex] / volume
            else:
                lastIndex = intervalIndex - 1
                if lastIndex != 0:
                    priceMatrix[dayIndex][intervalIndex] = priceMatrix[dayIndex][lastIndex]
    return priceMatrix


# 1.6 返回历史交易日的5分钟开盘价矩阵
def GetHistoryIntervalOpenPrice(stockId, startDay, endDay, timeSpan):
    fiveMinOpenPriceMatrix = [[0 for col in range(48)] for row in range(timeSpan)]
    global FiveMinOpenPriceMatrix
    TradeDayList = []
    calendar = md.get_calendar(Market, startDay, endDay)
    index = len(calendar) - 1
    for i in range(0, timeSpan):
        timeOri = calendar[index].strtime
        strIndex = timeOri.index('T')
        timeAft = timeOri[0:strIndex]
        TradeDayList.insert(0, timeAft)
        index -= 1
    for day in TradeDayList:
        dayIndex = TradeDayList.index(day)
        startTime = day + ' ' + '9:30:00'
        endTime = day + ' ' + '15:00:00'
        bars = md.get_bars(stockId, 300, startTime, endTime)
        for interval in range(len(bars)):
            fiveMinOpenPriceMatrix[dayIndex][interval] = bars[interval].open
    return fiveMinOpenPriceMatrix

# 讲矩阵平铺成数列
def ConvertMatrixToList(historyPriceMatrix):
    dataList = []
    rowsCount = len(historyPriceMatrix)
    for rowIndex in range(0, rowsCount):
        for col in historyPriceMatrix[rowIndex]:
            dataList.append(col)
    return dataList

# 将bar转换为开盘价
def ConvertBarsToOpenPrices(bars):
    openPriceList = []
    for bar in bars:
        openPriceList.append(bar.open)
    return openPriceList

# 将bar转换为成交均价
def ConvertBarsToAveragePricesList(bars):
    averagePriceList = []
    lastAveragePrice = 0
    for bar in bars:
        averagePrice = bar.amount / bar.volume
        if averagePrice != 0:
            averagePriceList.append(averagePrice)
        else:
            averagePriceList.append(lastAveragePrice)
        lastAveragePrice = averagePrice
    return averagePriceList
    
# 1.6 返回经过小波分解和拟合的结果
def GetWaveletHurst(historyPriceMatrix):
    dataList = []
    rowsCount = len(historyPriceMatrix)
    for rowIndex in range(0, rowsCount):
        for col in historyPriceMatrix[rowIndex]:
            dataList.append(col)
    print("dataList", dataList)
    coeffs = pywt.wavedec(dataList, 'db8', level=15)
    cA2, cD15, cD14, cD13, cD12, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    n15 = len(cD15)
    n14 = len(cD14)
    n13 = len(cD13)
    n12 = len(cD12)
    n11 = len(cD11)
    n10 = len(cD10)
    n9 = len(cD9)
    n8 = len(cD8)
    n7 = len(cD7)
    n6 = len(cD6)
    n5 = len(cD5)
    n4 = len(cD4)
    n3 = len(cD3)
    n2 = len(cD2)
    n1 = len(cD1)

    print("cD6:", cD6)
    print("cD6 rows", cD6.shape[0])
    print("cD5:", cD5)
    print("cD5 rows", cD5.shape[0])
    print("cD4:", cD4)
    print("cD4 rows", cD4.shape[0])
    print("cD3:", cD3)
    print("cD3 rows", cD3.shape[0])
    print("cD2:", cD2)
    print("cD2 rows", cD2.shape[0])
    print("cD1:", cD1)
    print("cD1 rows", cD1.shape[0])
    print("np.dot(cD6, cD6.T) ", np.dot(cD6.T, cD6))
    print("np.dot(cD5, cD5.T) ", np.dot(cD5.T, cD5))
    print("np.dot(cD4, cD4.T) ", np.dot(cD4.T, cD4))
    print("np.dot(cD3, cD3.T) ", np.dot(cD3.T, cD3))
    print("np.dot(cD2, cD2.T) ", np.dot(cD2.T, cD2))
    print("np.dot(cD1, cD1.T) ", np.dot(cD1.T, cD1))
    f15 = math.log(np.dot(cD15, cD15.T) / n15) / math.log(2)
    f14 = math.log(np.dot(cD14, cD14.T) / n14) / math.log(2)
    f13 = math.log(np.dot(cD13, cD13.T) / n13) / math.log(2)
    f12 = math.log(np.dot(cD12, cD12.T) / n12) / math.log(2)
    f11 = math.log(np.dot(cD11, cD11.T) / n11) / math.log(2)
    f10 = math.log(np.dot(cD10, cD10.T) / n10) / math.log(2)
    f9 = math.log(np.dot(cD9, cD9.T) / n9) / math.log(2)
    f8 = math.log(np.dot(cD8, cD8.T) / n8) / math.log(2)
    f7 = math.log(np.dot(cD7, cD7.T) / n7) / math.log(2)
    f6 = math.log(np.dot(cD6, cD6.T) / n6) / math.log(2)
    f5 = math.log(np.dot(cD5, cD5.T) / n5) / math.log(2)
    f4 = math.log(np.dot(cD4, cD4.T) / n4) / math.log(2)
    f3 = math.log(np.dot(cD3, cD3.T) / n3) / math.log(2)
    f2 = math.log(np.dot(cD2, cD2.T) / n2) / math.log(2)
    f1 = math.log(np.dot(cD1, cD1.T) / n1) / math.log(2)
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    f = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]
    cof = np.polyfit(t, f, 1)
    fit = np.polyval(cof, t)
    pl.plot(t, f, 'o')
    pl.plot(t, fit, '-')
    pl.show()

    L = (fit[15] - fit[0]) / 14
    H = (L + 1) / 2
    return H

def DrawTheData(MatrixData):
    x = []
    for i in range(1, 49):
        x.append(i)
    rowCount = len(MatrixData)
    for rowIndex in range(0, rowCount):
        sign = ['ro', 'bo', 'co', 'mo', 'yo', 'ko', 'r-', 'b-', 'c-', 'm-']
        pl.plot(x, MatrixData[rowIndex], sign[rowIndex])

def DrawTheDataForARow(listData, sign):
    x = []
    y = []
    for item in listData:
        x.append(item[0])
        y.append(item[1])
    pl.plot(x, y, sign)
    pl.show()


# 返回计算今天的各个时间片的权重（包装了1.1, 1.2, 1.3）
def GetRatioListOfToday(stockId, startDay, endDay, historyDays, weightList):
    FillVolumnMatrix(stockId, startDay, endDay, historyDays)
    vrMarix = FillVolumnRatioMatrix(VolumeMatrix)
    todayRatioList = CalTodayVolumnRatioList(vrMarix, weightList)
    return todayRatioList

# 2.1 返回下一个大时间片的订单数量
def CalNextLargeIntervalVol(remainVol, intervalNo):
    sumOfRemainRatio = 0
    for item in range(intervalNo, Intervals):
        sumOfRemainRatio = sumOfRemainRatio + TodayVRList[item]
    remainRio = TodayVRList[intervalNo] / sumOfRemainRatio
    currentVol = ToIntShare(remainVol * remainRio)
    return currentVol

# 2.2 将5分钟的订单量平均分配到5个1分钟时间片
def SplitLargeIntoOneMinAverage(vol, pieces=5):
    reminders = vol
    littleVolume = []
    for i in range(pieces):
        volume = ToIntShare(reminders / (pieces - i))
        reminders -= volume
        littleVolume.append(volume)
    return littleVolume

def ForecastNextPoint(priceDataList, pointCount=10):
    sum = 0
    dataCount = len(priceDataList)
    for index in range(dataCount - pointCount - 1,dataCount - 1):
        sum += priceDataList[index]
    averagePrice = sum / pointCount
    return averagePrice

# 2.3 currentIndex = 0时,
def GetCurrentPriceList(historyPriceMatrix, currentIndex=0):
    dataList = []
    rowsCount = len(historyPriceMatrix)
    for rowIndex in range(0, rowsCount):
        for col in historyPriceMatrix[rowIndex]:
            dataList.append(col)
    if currentIndex > 0:
        for index in range(0, currentIndex - 1):
            dataList.append(TradeDayPriceList[index])
    averagePrice = ForecastNextPoint(dataList, 10)
    dataList.append(averagePrice)
    return dataList

#返回一组当前交易日的预估值
def GetForcastAveragePricesListByMove(historyPriceList, currentPriceList):
    tempPriceList = copy.deepcopy(currentPriceList)
    for index in range(1, 6):
        indexitem = len(historyPriceList) - index
        tempPriceList.insert(0, historyPriceList[indexitem])
    forcastList = []
    totalCount = len(currentPriceList)
    weighList = [0.05, 0.1, 0.2, 0.25, 0.4]
    for index in range(0, totalCount):
        print("index", index)
        print("tempPriceList Count", len(tempPriceList))
        forcastItem = tempPriceList[index] * weighList[0] + tempPriceList[index+1] * weighList[1] + tempPriceList[index+2] * weighList[2] + tempPriceList[index+3] * weighList[3] + tempPriceList[index+4] * weighList[4]
        forcastList.append(forcastItem)
    return forcastList

#返回一组f，其中index从0开始,表示预测值的索引
def GetIFS(dataList, hurst):
    dataListCount = len(dataList)
    weighList = [0.05, 0.1, 0.2, 0.25, 0.4]
    averageData = 0
    for i in range(0, 5):
        averageData += dataList[dataListCount-5+i] * weighList[i]
    dataList.append(averageData)
    fNum = len(dataList)
    stateList = [[0 for col in range(0, 5)] for row in range(0, fNum-1)]
    aParam = 1 / fNum
    dScale = 1 / ((pow(aParam, 1 - hurst)) * fNum)
    for i in range(1, fNum):
        a = aParam
        e = (fNum * (i - 1)) / fNum
        c = (dataList[i] - dataList[i-1]) / fNum - dScale * (dataList[fNum-1]-dataList[0]) / fNum
        f = (fNum * dataList[i-1]) / fNum - dScale * (fNum * dataList[0]) / fNum
        stateList[i-1] = [a, e, c, f, dScale]
    return stateList

def TestIFS():
    barList = md.get_bars(Market+'.'+StockId, 300, CurrentDay + ' 9:40:00', CurrentDay + ' 10:40:00')
    averageDataList = []
    for i in range(0, len(barList)):
        vol = barList[i].volume
        amt = barList[i].amount
        if vol != 0:
            averagePrice = amt / vol
            averageDataList.append(averagePrice)
        else:
            tempCount = len(averageDataList)
            averageDataList.append(averageDataList[tempCount - 1])
    ifsTestList = GetIFS(averageDataList, 0.6)
    return ifsTestList

def Interpolation(ifsParameterList, ntimes):
    pointList = []
    pointList = Interation([[0,0]], ifsParameterList, ntimes)
    pointList = SortPoint(pointList)
    return pointList

def Interation(pointList, ifsParameterList, ntimes):
    if ntimes > 0:
        resultPointList = []
        pointCount = len(pointList)
        ifsCount = len(ifsParameterList)
        for i in range(0, pointCount):
            x = pointList[i][0]
            y = pointList[i][1]
            for j in range(0, ifsCount):
                a = ifsParameterList[j][0]
                e = ifsParameterList[j][1]
                c = ifsParameterList[j][2]
                f = ifsParameterList[j][3]
                d = ifsParameterList[j][4]
                L = a * x + e
                F = c * x + d * y + f
                resultPointList.append([L, F])
        print("ntimes", ntimes)
        print("resultPointList", resultPointList)
        ntimes -= 1
        return Interation(resultPointList, ifsParameterList, ntimes)
    else:
        return pointList

#按x坐标由小到大顺序排列点集
def SortPoint(pointList):
    outCount = len(pointList) - 1
    for i in range(0, outCount):
        for j in range(0, outCount-i):
            if (pointList[j][0] > pointList[j+1][0]):
                temp = pointList[j]
                pointList[j] = pointList[j+1]
                pointList[j+1] = temp
    return pointList

class Mystrategy(StrategyBase):
    def __init__(self, *args, **kwargs):
        super(Mystrategy, self).__init__(*args, **kwargs)


    def on_login(self):
        pass

    def on_error(self, code, msg):
        pass

    def on_bar(self, bar):
        pass
        '''timeStampNow = (pandas.Timestamp)(bar.strtime)
        global fiveMinQuantity
        global TargetQuantity
        global oneMinVolList
        if IntervalBorderList.count(timeStampNow) > 0:
            # 5分钟整点
            print(timeStampNow)
            intervalIndex = IntervalBorderList.index(timeStampNow)
            fiveMinQuantity = CalNextLargeIntervalVol(RemindQuantity, intervalIndex)
            oneMinVolList = SplitLargeIntoOneMinAverage(fiveMinQuantity, 5)
        minNow = timeStampNow.minute % 5
        td.open_short(Market, StockId, 0, oneMinVolList[minNow])'''


    def on_execrpt(self, res):
        pass

    def on_order_status(self, order):
        pass

    def on_order_new(self, res):
        pass

    def on_order_filled(self, res):
        global RemindQuantity
        RemindQuantity -= res.filled_volume

    def on_order_partiall_filled(self, res):
        global RemindQuantity
        RemindQuantity -= res.filled_volume

    def on_order_stop_executed(self, res):
        pass

    def on_order_canceled(self, res):
        pass

    def on_order_cancel_rejected(self, res):
        pass


if __name__ == '__main__':
    myStrategy = Mystrategy(
        username='309201566@qq.com',
        password='JINBOWEN1022',
        strategy_id='eec5f8e6-0f8c-11e7-b9a1-28d2443e5dda',
        subscribe_symbols=Subscribe_symbol,
        mode=4,
        td_addr='localhost:8001'
    )
    myStrategy.backtest_config(
        start_time=StartTime,
        end_time=EndTime,
        initial_cash=100000000000,
        transaction_ratio=1,
        commission_ratio=0.0001,
        slippage_ratio=0.001,
        price_type=0)

    testList = TestIFS()
    testInterPolationList = Interpolation(testList, 4)
    print("testInterPolationList Count:", len(testInterPolationList))
    print("testInterPolationList", testInterPolationList)
    DrawTheDataForARow(testInterPolationList, 'ro')
    '''barList = md.get_bars(Market + '.' + StockId, 300, CurrentDay + ' 10:40:00', CurrentDay + ' 10:45:00')
    averagePrice = barList[0].amount / barList[0].volume
    print("averagePrice", averagePrice)'''

    ret = myStrategy.run()
    print('exit code: ', ret)