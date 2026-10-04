Attribute VB_Name = "ExposureReportMacro"
Sub FormatExposureReport()
    Dim ws As Worksheet
    Set ws = ActiveWorkbook.Sheets("DoD Attribution")
    
    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    
    ws.Range("A1:E1").Font.Bold = True
    ws.Range("A1:E1").Interior.Color = RGB(0, 51, 102)
    ws.Range("A1:E1").Font.Color = RGB(255, 255, 255)
    
    Dim cell As Range
    For Each cell In ws.Range("D2:D" & lastRow)
        If Abs(cell.Value) > 100000 Then
            cell.Interior.Color = RGB(255, 153, 153)
            cell.Font.Bold = True
            cell.Font.Color = RGB(153, 0, 0)
        End If
    Next cell
    
    ws.Columns("A:E").AutoFit
    MsgBox "Report Formatted and Exceptions Highlighted successfully.", vbInformation
End Sub
