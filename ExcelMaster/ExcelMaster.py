__Author__ = "Jingkai SUN"
__Date__ = "2025.05.15"

import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell, xl_range, xl_cell_to_rowcol
import openpyxl
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings

from .ExcelFormatTool import ExcelFormat
from .Utility import *

class ExcelWorkbook(ExcelFormat):
    """ Write anything to Excel (Workbook-level Operator). """
    def __init__(self, filepath, verbose = True):
        super().__init__(filepath)
        self.verbose = verbose
        self.ws_dict = {}

    def to_cell_range_text(self, first_row, first_col, last_row, last_col):
        """ To Excel Cell Range. """
        return xl_range(first_row, first_col, last_row, last_col)

    def cell_range_to_loc(self, cell_range_text):
        """ Convert text cell range expression to a list of values. """
        cell_range_list = cell_range_text.split(":")
        res = []
        for x in cell_range_list:
            res += list(xl_cell_to_rowcol(x))
        return res

    def colletter_to_textloc(self, row_index, col_letter):
        """ Append Row Index to the given Column Letter-Formatted Index. """
        col_letter = col_letter if ":" in col_letter else (col_letter + ":" + col_letter)
        output = col_letter.split(":")
        output = ":".join([col + str(row_index) for col in output])
        return output

    def set_color_scale(self, worksheet, cell_range,
                        colors = ("#F8696B", "#FFEB84", "#63BE7B")):
        """ Set Color Scale. """
        if len(colors) == 3:
            f = {
                    "type": "3_color_scale",
                    "min_color": colors[0],
                    "mid_color": colors[1],
                    "max_color": colors[2],
                }
        elif len(colors) == 2:
            f = {
                    "type": "2_color_scale",
                    "min_color": colors[0],
                    "max_color": colors[1],
                }
        else:
            raise ValueError("Please give 2 or 3 colors!")

        if isinstance(cell_range, list) or isinstance(cell_range, tuple):
            cell_range = self.to_cell_range_text(*cell_range)

        worksheet.conditional_format(cell_range, f)
        return 0

    def set_data_bar(self, worksheet, cell_range, bar_color = "#63C384"):
        """ Set Data Bar in a Worksheet for a range of cells. """
        
        f = {'type': 'data_bar',
             'data_bar_2010': True,
             "bar_color": bar_color}

        if isinstance(cell_range, list) or isinstance(cell_range, tuple):
            cell_range = self.to_cell_range_text(*cell_range)

        worksheet.conditional_format(cell_range, f)
        return 0

    def set_cell_format(self, worksheet, cell_range, cformat, cell_condition = None):
        """ Set format for a range of cell. """

        if cell_condition and isinstance(cell_condition, tuple):
            criteria = cell_condition[0].lower()
            value = cell_condition[1]
        
        if isinstance(cell_range, list) or isinstance(cell_range, tuple):
            cell_range = self.to_cell_range_text(*cell_range)

        if isinstance(cformat, str):
            cformat = self.dict_cell_format[cformat]
    
        if cell_condition and isinstance(cell_condition, tuple):
            
            if (criteria == 'between') or (criteria == 'not between'):
                worksheet.conditional_format(cell_range, {"type": "cell", 
                                                          "criteria": criteria, 
                                                          "minimum": value[0],
                                                          "maximum": value[1], 
                                                          "format": cformat})
    
                return 0
                
            worksheet.conditional_format(cell_range, {"type": "cell", 
                                                      "criteria": criteria, 
                                                      "value": value, 
                                                      "format": cformat})
            return 0
        
        worksheet.conditional_format(cell_range, {'type': 'no_errors', "format": cformat}) 
        return 0

    def set_cell_format_rbyr(self, worksheet, start_row, condition_list, condition, ifelse_col, cformat = "YELLOW_BG"):
        """ Conditionally Highlight Cells Row by Row. """
        
        if_col = ifelse_col[0]
        else_col = ifelse_col[1]
    
        i = 0
        row_concur = start_row
        while i < len(condition_list):
            target = condition_list[i]
        
            if val_input_condition(target, condition):
                if if_col:
                    self.set_cell_format(worksheet, 
                                       cell_range=self.colletter_to_textloc(row_concur + 1, if_col), 
                                       cformat=cformat)
                
            else:
                if else_col:
                    self.set_cell_format(worksheet, 
                                       cell_range=self.colletter_to_textloc(row_concur + 1, else_col), 
                                       cformat=cformat)
                
            i += 1
            row_concur = row_concur + 1
        return 0

    def remove_tmp_img(self, img_pattern = r".tmp_image_[0-9]+.png"):
        """ Remove Temp Images. """
        tmp_imgs = list_files(location = "./", pattern = img_pattern)
        if len(tmp_imgs) > 0:
            for f in tmp_imgs:
                os.remove(f)
            return 0
        return 

    def close_workbook(self):
        '''
        Finish writing the contents of the workbook and close the file.
        '''
        self.workbook.close()
        self.remove_tmp_img(img_pattern = r".tmp_image_[0-9]+.png")
        logging.info("All temp images have been removed.")


class ExcelMaster(ExcelWorkbook):
    """  Write anything to Excel (Worksheet-level Operator) """
    def __init__(self, filepath, verbose, gap_number = 2, init_loc = (0, 0)):
        super().__init__(filepath, verbose)
        self.curr_row = init_loc[0]
        self.curr_col = init_loc[1]
        self.gap_number = gap_number + 1

        self.default_row_height = 20
        self.default_col_width = 64

        self.max_nrows = 1048576
        self.max_ncols = 16384

    def add_worksheet(self, name, hide_grid = True, reset_loc = True, cell_scale = True, auto_fit = False, zoom_perc = 100, tab_color = None):
        """ Add a worksheet. """
        ws = self.workbook.add_worksheet(name)
        
        if hide_grid:
            ws.hide_gridlines(2)
            
        if reset_loc:
            self.reset_curr_loc()
            
        if isinstance(cell_scale, tuple):
            self.set_cell_size(ws, cell_scale)
        if cell_scale is True:
            self.set_cell_size(ws)
            
        if auto_fit:
            ws.auto_fit()

        if tab_color:
            ws.set_tab_color(tab_color)
            
        ws.set_zoom(zoom_perc)
        self.ws_dict[name] = ws
        self.engine.sheets[name] = ws
        return ws

    def reset_curr_loc(self, loc = (0, 0)):
        """ Reset Current Location. """
        
        self.curr_row = loc[0]
        self.curr_col = loc[1]
        return 0

    def _reset_cell_size(self):
        """ Reset Cell Size to Default Value. """
        self.default_row_height = 20
        self.default_col_width = 64
        return 0

    def set_cell_size(self, worksheet, size_scale = (1, 1)):
        """ Set Cell Size in Scale. """
        
        if isinstance(size_scale, tuple) and len(size_scale) == 2:
            self._reset_cell_size()
            self.default_row_height = self.default_row_height * size_scale[0]
            self.default_col_width =  self.default_col_width * size_scale[1]

        for i in range(0, self.max_nrows):
            worksheet.set_row_pixels(i, height=self.default_row_height)
            
        worksheet.set_column_pixels(0, self.max_ncols - 1, width=self.default_col_width)
        return 0

    def get_curr_loc(self, toCell = False):
        """ Get Current Location in worksheet. """
        if toCell:
            return xl_rowcol_to_cell(self.curr_row, self.curr_col)
        return (self.curr_row, self.curr_col)
    
    def set_border_line(self, worksheet, valuerange, border_line = 1):
        """ Set border line for a range of cells. """
        border_fmt = self.workbook.add_format({'bottom': border_line, 'top': border_line, 'left': border_line, 'right': border_line})
        self.set_cell_format(worksheet = worksheet, cell_range = valuerange, cformat = border_fmt)
        return 0

    def merge_col(self, worksheet, loc = None, nrows = 1, ncols = 1, text = "", skipby = 'row', cformat = 'BLUE_H4', retCellRange = None):
        """ Merge columns in a single row. """
        
        start_row = loc[0] if loc else self.curr_row
        start_col = loc[1] if loc else self.curr_col
        written_range = [start_row, start_col, start_row + nrows - 1, start_col + ncols - 1]
        
        worksheet.merge_range(*written_range, text, self.dict_cell_format[cformat])

        if self.verbose:
            logging.info(f"Merged Cells: {self.to_cell_range_text(*written_range)}")

        # Skipped by Rows/Columns
        if skipby == 'row':
            self.curr_row = (start_row + nrows + self.gap_number)
        if skipby == 'col':
            self.curr_col = (start_col + ncols + self.gap_number)

        # Return written location by Cell Text/Value Range
        if retCellRange == "text":
            return self.to_cell_range_text(*written_range)
        if retCellRange == "value":
            return written_range
            
        return 0

    def write_dataframe(self, worksheet, df, loc = None, title = None, index = False, header = True, skipby = 'row', titleformat = "BLUE_H4", headerformat = "TABLE_HEADER", valueformat="----", retCellRange = None):
        """ Write a dataframe to excel file. """
        
        start_row = loc[0] if loc else self.curr_row
        start_col = loc[1] if loc else self.curr_col

        ncols = df.shape[1] 
        nrows = df.shape[0]

        ## Get Number of Index Columns
        index_ncols = 0
        if index:
            index_ncols = len(df.index.names)
            ncols += index_ncols

        ## Get Number of Header Rows
        header_nrows = 0
        if header:
            header_nrows = len(df.columns.names)
            nrows += header_nrows

        ## Get Number of Title Rows
        title_nrows = 0
        if title:
            title_nrows = 1
            nrows += title_nrows

        ## Get Header Range (Include Index)
        if header:
            header_start_row = (start_row + title_nrows)
            header_start_col = start_col
            header_end_row = max((start_row + header_nrows - 1), header_start_row)
            header_end_col = (start_col + ncols - 1)
            header_range = self.to_cell_range_text(header_start_row, header_start_col, 
                                                   header_end_row, header_end_col)

        ## Write DataFrame
        if title:
            self.merge_col(worksheet = worksheet, loc = loc, ncols = ncols, cformat=titleformat, 
                           text = title, skipby = None)
            df.to_excel(self.engine, sheet_name = worksheet.name, startrow = start_row + title_nrows, 
                        startcol = start_col, header = header, index = index)
        else:
            df.to_excel(self.engine, sheet_name = worksheet.name, startrow = start_row, 
                        startcol = start_col, header = header, index = index)


        ## Get Value Range (Include Index)
        value_start_row = (start_row + title_nrows + header_nrows)
        value_start_col = start_col
        value_end_row = (start_row + nrows - 1)
        value_end_col = (start_col + ncols - 1)
        value_range = [value_start_row, value_start_col, value_end_row, value_end_col]

        ## Set Format
        self.set_cell_format(worksheet = worksheet, cell_range = value_range, cformat = valueformat)
        if header:
            self.set_cell_format(worksheet = worksheet, cell_range = header_range, cformat = headerformat)

        written_range = [start_row, start_col, (start_row + nrows - 1), (start_col + ncols - 1)]
        
        if self.verbose:
            logging.info(f"Table Written in Cell Range: {self.to_cell_range_text(*written_range)}")
        
        if skipby == 'row':
            self.curr_row = (start_row + nrows + self.gap_number)
        if skipby == 'col':
            self.curr_col = (start_col + ncols + self.gap_number)

        if retCellRange == "text":
            return self.to_cell_range_text(*written_range)
        if retCellRange == "value":
            return written_range
            
        return 0
        
    def _get_image_size(self, figPath, figScale = (1, 1), retSizeInCell = True):
        """ Get Image Size in Excel Cells. """
        img = Image.open(figPath)
        w, h = img.size
        (width, height) = (img.width * figScale[0], img.height * figScale[1])
        h_in_cell = np.floor(height/self.default_row_height)
        w_in_cell = np.ceil(width/self.default_col_width)
        img.close()
        if retSizeInCell:
            return (w_in_cell, h_in_cell)
        return (width, height)

    def _resize_image(self, imgPath, resize, outPath, size_in_cell = True):
        """ Resize image. """
        # Open the image
        image = Image.open(imgPath)
        new_image = image.resize((resize[0], resize[1]))
        if size_in_cell:
            # Resize the image
            new_image = image.resize((resize[1] * self.default_col_width, resize[0] * self.default_row_height))
        # Save the resized image
        new_image.save(outPath)
        image.close()
        return 0
    
    def insert_image(self, worksheet, figPath, figScale = (1, 1), loc = None, skipby = 'row', retCellRange = None):
        """ Insert an image to the sheet. """
        start_row = loc[0] if loc else self.curr_row
        start_col = loc[1] if loc else self.curr_col

        start_cell = xl_rowcol_to_cell(start_row, start_col)
        figsize_in_cells = self._get_image_size(figPath = figPath, figScale = figScale, retSizeInCell = True)

        worksheet.insert_image(start_cell, figPath, {"x_scale": figScale[0], "y_scale": figScale[1]})

        # Skipped by Rows/Columns
        if skipby == 'row':
            self.curr_row = (start_row + int(figsize_in_cells[1]) + self.gap_number)
        if skipby == 'col':
            self.curr_col = (start_col + int(figsize_in_cells[0]) + self.gap_number)

        written_range = [start_row, start_col, 
                         start_row + int(figsize_in_cells[1]), 
                         start_col + int(figsize_in_cells[0])]
        
        # Return written location by Cell Text/Value Range
        if retCellRange == "text":
            return self.to_cell_range_text(*written_range)
        if retCellRange == "value":
            return written_range
            
        if self.verbose:
            logging.info(f"Image Written in Cell Range: {self.to_cell_range_text(*written_range)}")
            
        return 0

    def __add_chart_data_tab(self, tabname, hide = True, max_num = 99999):
        """ Add a temp tab for chart data. """
        i = 0
        while i <= max_num:
            tabname_i = (tabname + str(i))
            if tabname_i not in self.ws_dict:
                ws = self.add_worksheet(tabname_i, reset_loc = False, 
                                        cell_scale = None, 
                                        auto_fit = False, 
                                        zoom_perc = 100)
                ws.hide()
                return ws
            i += 1
        return -1

    def _transpose_df_for_chart(self, df, y_list, x = None):
        """ Transpose Dataframe for Chart Structrue. """

        _df = df.copy()
        _df.columns.name = None
        _df.index.name = None
        
        if x is None:
            _df = _df[y_list].T
            _df = _df.reset_index(drop=False)
        else:
            x = x if isinstance(x, list) else [x]
            _df = tanspose_dataframe(_df[[*x, *y_list]], x)
            
        _df.columns.name = None
        _df.index.name = None
        return _df

    def __convert_to_none_tuple(self, x = None):
        """ Convert None Type to Tuple of Nones. """
        if x is None:
            return (None, None)
        return x

    def __validate_input_chart_obj(self, chart_type, input_chart):
        """ Validate if the given append_to_chart object is aligned with chart_type argument. """

        if chart_type == "line":
            return isinstance(input_chart, xlsxwriter.chart_line.ChartLine)
        if chart_type in ["column", "stacked_column"]:
            return isinstance(input_chart, xlsxwriter.chart_column.ChartColumn)
        if chart_type == "pie":
            return isinstance(input_chart, xlsxwriter.chart_pie.ChartPie)

    def write_chart(self, worksheet, df, y_list, x = None, title = "", 
                    chart_size = (30, 13), chart_type = "line",
                    y_axis_range = (None, None), y_num_format = None,
                    y2_axis = False, loc = None, retChart = False, 
                    skipby = "row", outputData = False, retCellRange = None,
                    xy_axes_name = ("", ""), major_gridlines = False,
                    legend = "bottom", line_marker = "circle", line_type = "solid", 
                    chart_style = None, append_to_chart = None): 
        """ Write line chart to Excel worksheet."""

        if isinstance(chart_type, str):
            if chart_type == "line":
                chart_type = {'type': 'line'}
            elif chart_type == "column":
                chart_type = {'type': 'column'}
            elif chart_type == "stacked_column":
                chart_type = {'type': 'column', "subtype": "stacked"}
            elif chart_type == "pie":
                chart_type = {"type": "pie"}
            else:
                raise ValueError(" Please select one from 'line', 'column', 'stacked_column' or 'pie'. ")

        start_row = loc[0] if loc else self.curr_row
        start_col = loc[1] if loc else self.curr_col

        y_axis_range = self.__convert_to_none_tuple(y_axis_range)

        df_t = self._transpose_df_for_chart(df = df, y_list = y_list, x = x)
        if x is None:
            x = list(df.index.names)
        if isinstance(x, str):
            x = [x]
        y = "index" if len(x) == 1 else tuple(["index"] + [''] * (len(x) - 1))
        x_list = [x for x in df_t.columns if x != y]

        data_nrows = df_t.shape[0]
        data_ncols = df_t.shape[1]

        raw_data_ws = self.__add_chart_data_tab("__CHRT_DATA_")

        ## if y not given, then use index value as y series.
        data_index_as_y = True
        if y is None:
            sel_df = df_t[[*x_list]]
        else:
            sel_df = df_t[[y, *x_list]]
            sel_df = sel_df.set_index(y)
            sel_df.index.name = None

        ## if output chart data to current worksheet.
        if outputData:
            raw_data_ws = worksheet
            df_range = self.write_dataframe(worksheet = raw_data_ws, df = sel_df, index = data_index_as_y, title = None, skipby=None, retCellRange="value", loc = (start_row, start_col))
            chart_loc = (df_range[2] + self.gap_number, df_range[1])
        else:
            df_range = self.write_dataframe(worksheet = raw_data_ws, df = sel_df, index = data_index_as_y, title = None, skipby=None, retCellRange="value", loc = (0,0))
            chart_loc = (start_row, start_col)
        
        x_count = len(x_list)
        nrows = df_t.shape[0]

        data_row_anchor = df_range[0]
        data_col_anchor = df_range[1]
        row_shift = 0

        category_range = [df_range[0], df_range[1] + 1, 
                          df_range[0] + (len(x) - 1), df_range[3]]
        
        if self.verbose:
            logging.info(f"Category Cell Range: {self.to_cell_range_text(*category_range)}")

        chart = self.workbook.add_chart(chart_type)
        if append_to_chart is not None:
            if self.__validate_input_chart_obj(chart_type = chart_type["type"], input_chart = append_to_chart):
                chart = append_to_chart
            else:
                warnings.warn("WARNING: Can only append data to the chart that has the same chart type as your given one.")

        value_start = len(x) + 1 if len(x) > 1 else len(x)
        for row_shift in range(value_start, value_start + nrows, 1):

            value_range = [data_row_anchor + row_shift, data_col_anchor + 1, 
                           data_row_anchor + row_shift, data_col_anchor + x_count]
            
            chart.add_series({
                'name':        [raw_data_ws.name, (data_row_anchor + row_shift), data_col_anchor],
                'categories':  [raw_data_ws.name, *category_range],
                'values':      [raw_data_ws.name, *value_range],
                'marker':      {'type': line_marker},
                "line" :       {'dash_type': line_type},
                'data_labels': {'percentage': True} if chart_type['type'] == 'pie' else None,
                'y2_axis':     y2_axis,
            })
            if self.verbose:
                logging.info("Values Cell Range: %s", self.to_cell_range_text(*value_range))
                
        chart.set_title({'name': title})
        chart.set_legend({'position': str(legend)})
        chart.set_size({"height": chart_size[0] * self.default_row_height, # by row 15
                        "width": chart_size[1] * self.default_col_width}) # by col 8.43

        chart.set_x_axis({"name": xy_axes_name[0]})

        set_y_axis_dict = {"name": xy_axes_name[1], 
                           "major_gridlines": {"visible": int(major_gridlines)},
                           'min': y_axis_range[0], 
                           'max': y_axis_range[1],
                           'num_format': y_num_format}
        if y2_axis:
            chart.set_y2_axis(set_y_axis_dict)
        else:
            chart.set_y_axis(set_y_axis_dict)

        # Setting Chart Style
        if chart_style:
            chart.set_style(chart_style)

        # Need to Return Chart before Insert it to the worksheet.
        if retChart:
            return chart
        
        worksheet.insert_chart(
            chart_loc[0], 
            chart_loc[1], 
            chart
        )

        written_range = [start_row, start_col, int(chart_loc[0] + chart_size[0]), int(chart_loc[1] + chart_size[1])]
        if outputData:
            
            written_range = [start_row, start_col, 
                             (start_row + data_nrows + self.gap_number + chart_size[0]),
                             max(df_range[3], start_col + chart_size[1])]

        if skipby == 'row':
            if outputData:
                self.curr_row = (start_row + data_nrows + self.gap_number + chart_size[0] + self.gap_number - 1)
            else:
                self.curr_row = (start_row + chart_size[0] + self.gap_number - 1)
        if skipby == 'col':
            self.curr_col = (start_col + int(chart_size[1]) + self.gap_number)

        if self.verbose:
            logging.info(f"Chart Written in Cell Range: {self.to_cell_range_text(*written_range)}")

        if retCellRange == "text":
            return self.to_cell_range_text(*written_range)
        if retCellRange == "value":
            return written_range
        
        return 0

    def write_combined_chart(self, worksheet, chart1, chart2, 
                             loc = None, chart_size = (30, 13), 
                             skipby = "row", retCellRange = None):
        """ Combined two chart objects and then write to Excel worksheet. """
        
        start_row = loc[0] if loc else self.curr_row
        start_col = loc[1] if loc else self.curr_col
        
        chart1.combine(chart2)
        worksheet.insert_chart(xl_rowcol_to_cell(start_row, start_col), chart1)

        written_range = [start_row, start_col, (start_row + chart_size[0]), (start_col + chart_size[1])]

        if skipby == 'row':
            self.curr_row = (start_row + chart_size[0] + self.gap_number - 1)
        if skipby == 'col':
            self.curr_col = (start_col + chart_size[1] + self.gap_number)

        if self.verbose:
            logging.info(f"Chart Written in Cell Range: {self.to_cell_range_text(*written_range)}")

        if retCellRange == "text":
            return self.to_cell_range_text(*written_range)
        if retCellRange == "value":
            return written_range

        return 0
        

    def write_duo_chart(self, worksheet, df,
                        y1_list, y2_list = None, x = None,
                        c1_type = "column", c2_type = "line", 
                        y1_axis_range = (0, 1), y2_axis_range = None,
                        y1_num_format = None, y2_num_format = None,
                        y1_line_marker = "circle", y2_line_marker = "circle",
                        y1_line_type = "solid", y2_line_type = "solid",
                        loc = None, title = "", chart_size = (30, 13),
                        xy_axes_name = ("", ""), major_gridlines = False,
                        retChart = False, retCellRange = None,
                        skipby = "row"):
        """ Write duo-chart sharing the same x axis. """

        start_row = loc[0] if loc else self.curr_row
        start_col = loc[1] if loc else self.curr_col

        y2_axis = True
        if y2_list is None:
            y2_axis = False

        if y2_axis_range is None:
            y2_axis_range = y1_axis_range

        xy_name = (xy_axes_name[0], xy_axes_name[1])
        xy2_name = (xy_axes_name[0], xy_axes_name[1])
        if y2_axis and len(xy_axes_name) >= 3:
            xy_name = (xy_axes_name[0], xy_axes_name[1])
            xy2_name = (xy_axes_name[0], xy_axes_name[2])
        if y2_axis and len(xy_axes_name) < 3:
            xy_name = (xy_axes_name[0], xy_axes_name[1])
            xy2_name = (xy_axes_name[0], "")
            
        chart1 = self.write_chart(df = df, 
                                  x = x, 
                                  y_list = y1_list, 
                                  worksheet = worksheet, 
                                  title = title,
                                  chart_type = c1_type, 
                                  chart_size = chart_size,
                                  y_axis_range = y1_axis_range,
                                  xy_axes_name = xy_name,
                                  major_gridlines = major_gridlines,
                                  y_num_format = y1_num_format,
                                  line_type = y1_line_type,
                                  line_marker = y1_line_marker,
                                  retChart = True)
        
        chart2 = self.write_chart(df = df, 
                                  x = x,
                                  y_list = y2_list, 
                                  worksheet=worksheet, 
                                  title = title, 
                                  chart_type = c2_type, 
                                  chart_size = chart_size,
                                  y_axis_range = y2_axis_range,
                                  y2_axis = y2_axis,
                                  xy_axes_name = xy2_name,
                                  major_gridlines = False,
                                  y_num_format = y2_num_format,
                                  line_type = y2_line_type,
                                  line_marker = y2_line_marker,
                                  retChart=True)

        # Need to Return Chart before Insert it to the worksheet.
        if retChart:
            return (chart1, chart2)

        cell_range = self.write_combined_chart(worksheet, 
                                  chart1, chart2, 
                                  loc = loc, chart_size = chart_size, 
                                  skipby = skipby, retCellRange = retCellRange)
            
        return cell_range

    def write_text_by_dict(self, worksheet, dict_cells):
        """ Write text using Python dictionary. """
        for cell in dict_cells:
            list_contents = dict_cells[cell]
            
            if ':' in cell:
                text, cell_format = list_contents
                worksheet.merge_range(cell, text, self.dict_cell_format[cell_format])
            
            else:
                if len(list_contents) == 1:
                    list_items = []

                    # Text formats begin with '~~~'
                    list_mixed_items = list_contents[0]
                    for item in list_mixed_items:
                        if item.startswith('~~~'):
                            cell_format = item.split('~~~')[1]
                            item = self.dict_cell_format[cell_format]
                            
                            list_items.append(item)
                        else:
                            list_items.append(item)

                    worksheet.write_rich_string(cell, *list_items)

                else:
                    text, cell_format = list_contents
                    worksheet.write(cell, text, self.dict_cell_format[cell_format])

    def __split_line_by_format_sign(self, line):
        """ To split line by specified format sign '{}'. """

        ## find all format sign '{}'
        format_sign = re.findall(".*?{(.*?)}.*?", line)
        format_sign = format_sign if len(format_sign) != 0 else ['']
        format_sign = [x.lstrip().rstrip() for x in format_sign]

        ## find all cell formats
        cell_format_sign = re.findall(".*?\[\[(.*?)\]\]$", line.strip().replace("\n", ""))
        cell_format_sign = [x.lstrip().rstrip() for x in cell_format_sign]
        
        ## clean up text by removing cell format express
        if len(cell_format_sign) > 0:
            line = line.split("[[")[0]

        ## find text before the first format sign appeared.
        text_bf_curly = re.findall(r"(.*?)\{", line)
        text_bf_curly = [text_bf_curly[0]] if len(text_bf_curly) != 0 else ['']
        text_bf_curly = [x.strip("\n") for x in text_bf_curly]

        ## find text after the first format sign appeared.
        text_af_curly = re.findall(r"\}\s*(.*?)(?=\s*\{|$)", line) if re.search(".*?{(.*?)}.*?", line) else [line]
        text_af_curly = [x.strip("\n") for x in text_af_curly]

        return text_bf_curly, format_sign, text_af_curly, cell_format_sign

    def __parse_line_by_format_sign(self, worksheet, loc, line):
        """ Parse text by format sign. """
        
        text_bf_curly, format_sign, text_af_curly, cell_format_sign = self.__split_line_by_format_sign(line)
        
        start_text = text_bf_curly[0]
        # if len(start_text) > 0 or len(format_sign) >= 2:
        if len(format_sign) >= 2:
            # if one line has more than 2 formats specified.
            
            res = [loc, start_text] if start_text != '' else [loc]
            assert len(format_sign) == len(text_af_curly)
            
            for fmt, s2 in zip(format_sign, text_af_curly):
                res.append(self.dict_cell_format[fmt] if fmt else self.dict_cell_format[""])
                res.append(s2.strip('\n')) # remove enter sign
                
            # Write line by line
            if self.verbose:
                logging.info(res)
                
            worksheet.write_rich_string(*res)
    
        if len(format_sign) < 2:
            # if one line has more than 2 formats specified.
            
            res = [loc, start_text] if start_text != '' else [loc]
            assert len(format_sign) == len(text_af_curly)
            
            fmt = format_sign[0]
            text = text_af_curly[0].strip("\n")
            res.append(text)
            res.append(self.dict_cell_format[fmt])

            # Write line by line
            if self.verbose:
                logging.info(res)
                
            worksheet.write_string(*res)

        if len(cell_format_sign) > 0:
            self.set_cell_format(worksheet = worksheet, cell_range = loc, cformat = cell_format_sign[0])
        return 0

    def write_text_content(self, worksheet, input_text = None, txt_path = None, loc = None, retCellRange = None):
        """ Write text content line by line to the worksheet. """
        
        row_anchor = loc[0] if loc else self.curr_row
        col_anchor = loc[1] if loc else self.curr_col

        start_row = row_anchor
        start_col = col_anchor
        
        if (input_text) and (txt_path):
            raise ValueError("Please give either input_text or txt_path, not both !!!")
        
        if (input_text is None) and (txt_path is None):
            raise ValueError("Please specify either input_text or txt_path.")
        
        if txt_path is not None:
            file_notes = open(txt_path, 'r')
            textlines = file_notes.readlines()
        
        if input_text is not None:
            textlines = input_text.split("\n")
        
        for line in textlines:
            # Write text line by line
            loc = xl_rowcol_to_cell(row_anchor, col_anchor)

            if " [>] " in line:
                line_split = line.split(" [>] ")
                col_shift = 0
                for sub_line in line_split:
                    sub_line = sub_line.rstrip()
                    self.__parse_line_by_format_sign(worksheet = worksheet, loc = loc, line = sub_line)
                    col_shift += 1
                    loc = xl_rowcol_to_cell(row_anchor, col_anchor + col_shift)
            else:
                self.__parse_line_by_format_sign(worksheet = worksheet, loc = loc, line = line)
            
            row_anchor += 1
        
        self.curr_row = (row_anchor - 1)

        end_row = row_anchor
        end_col = col_anchor

        written_range = [start_row, start_col, end_row, end_col]
        
        if self.verbose:
            logging.info(f"Text Written in Cell Range: {self.to_cell_range_text(*written_range)}")
            
        if retCellRange == "text":
            return self.to_cell_range_text(*written_range)
        if retCellRange == "value":
            return written_range
        
        return 0

    @staticmethod
    def plot_boxplot(df, x, y, y_percentage = False, colored_box = True, color_grp = (10, 1), title = "", fontsize = 14, figsize = (8, 6), show_fig = True, img_path = None, transp_bg = False):
        """ Plot Box Plot Chart using matplotlib. """
        
        plt.style.use('default')
        
        fig, ax = plt.subplots(1, 1, figsize = figsize, dpi=200)
        
        box_plot_data = convert_to_boxplot_data(df, x, y, True)
        
        bplot = ax.boxplot(box_plot_data.values(), patch_artist=colored_box, widths = 0.4)
        ax.set_xticklabels(box_plot_data.keys(), fontsize = fontsize)
        ax.set_ylabel(string_proc(y), color='black', fontsize = fontsize)
        ax.set_xlabel(string_proc(x), color='black', fontsize = fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        ax.set_title(title, fontsize=fontsize + 4)
    
        if colored_box:
            colors = color_input_validation(color_grp, len(box_plot_data.keys()))
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(.7)
        
        if y_percentage:
            plt.gca().yaxis.set_major_formatter(PercentFormatter(100)) 
        plt.tight_layout()
        
        if show_fig:
            plt.show()
        
        if img_path:
            fig.savefig(img_path, transparent = transp_bg, dpi=200)
        
        plt.close()
        return fig, ax

    def write_boxplot(self, ws, df, x, y, y_percentage = False, colored_box = True, color_grp = (10, 1), title = "", 
                      fontsize = 14, figsize = (30, 13), show_fig = False, img_path = None, transp_bg = False,
                      loc = None, skipby = "row", retCellRange = None):
        """ Write boxplot to Worksheet. """
        
        currTime = getCurrentDateTime()
        rn = str(random.randrange(0, 1000))
        rn_num = currTime + rn
        tmp_image_path = f"./.tmp_image_{rn_num}.png"
        
        ExcelMaster.plot_boxplot(df = df, 
                                 x = x, y = y, 
                                 y_percentage = y_percentage, 
                                 colored_box = colored_box, 
                                 color_grp = color_grp, 
                                 title = title, 
                                 fontsize = fontsize, 
                                 show_fig = show_fig, 
                                 transp_bg = transp_bg,
                                 img_path = tmp_image_path)

        self._resize_image(tmp_image_path, figsize, tmp_image_path)
        ret_range = self.insert_image(ws, figPath=tmp_image_path, figScale=(1, 1), loc = loc, skipby = skipby, retCellRange = retCellRange)
        
        return ret_range