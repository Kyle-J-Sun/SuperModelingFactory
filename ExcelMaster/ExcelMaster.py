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
    """Write anything to Excel (Workbook-level Operator).

    Adds workbook-level helpers (conditional formats, color scales, data bars, cell range conversion and clean-up of
    temporary images) on top of ``ExcelFormat``, which creates the workbook and the preset cell formats and provides the
    attributes ``engine``, ``workbook``, ``basename``, ``base_filepath`` and ``dict_cell_format``. The worksheet-level
    writing API is in the subclass ``ExcelMaster``.

    Parameters
    ----------
    filepath : str
        Path of the output ``.xlsx`` file. The file is created when the object is built and filled by ``close_workbook``.
    verbose : bool, default True
        If True, the writing methods log the cell range they wrote through ``logging.info``.

    Attributes
    ----------
    verbose : bool
        The ``verbose`` argument; it can be changed at any time.
    ws_dict : dict
        Worksheets by name, ``{name: xlsxwriter worksheet}``. It is filled by ``ExcelMaster.add_worksheet`` and also holds
        the hidden chart data sheets ``__CHRT_DATA_<N>``.
    """
    def __init__(self, filepath, verbose = True):
        """Create the workbook for ``filepath`` and set ``verbose``.

        The parameters and attributes are described in the class docstring.
        """
        super().__init__(filepath)
        self.verbose = verbose
        self.ws_dict = {}

    def to_cell_range_text(self, first_row, first_col, last_row, last_col):
        """To Excel Cell Range.

        Parameters
        ----------
        first_row : int
            Zero-based index of the first row.
        first_col : int
            Zero-based index of the first column.
        last_row : int
            Zero-based index of the last row.
        last_col : int
            Zero-based index of the last column.

        Returns
        -------
        str
            The range in A1 notation, for example ``"A1:C3"`` for ``(0, 0, 2, 2)``.
        """
        return xl_range(first_row, first_col, last_row, last_col)

    def cell_range_to_loc(self, cell_range_text):
        """Convert text cell range expression to a list of values.

        Parameters
        ----------
        cell_range_text : str
            Range in A1 notation (``"A1:C3"``) or a single cell (``"B2"``).

        Returns
        -------
        list of int
            Zero-based ``[first_row, first_col, last_row, last_col]`` for a range, or ``[row, col]`` for a single cell. It is
            the inverse of ``to_cell_range_text``.
        """
        cell_range_list = cell_range_text.split(":")
        res = []
        for x in cell_range_list:
            res += list(xl_cell_to_rowcol(x))
        return res

    def colletter_to_textloc(self, row_index, col_letter):
        """Append Row Index to the given Column Letter-Formatted Index.

        Parameters
        ----------
        row_index : int
            Excel row number (one-based, as shown in Excel) appended to the column letter(s).
        col_letter : str
            A column letter such as ``"B"`` or a column letter range such as ``"B:D"``.

        Returns
        -------
        str
            A one-row range in A1 notation: ``"B6:B6"`` for ``(6, "B")`` and ``"B6:D6"`` for ``(6, "B:D")``.
        """
        col_letter = col_letter if ":" in col_letter else (col_letter + ":" + col_letter)
        output = col_letter.split(":")
        output = ":".join([col + str(row_index) for col in output])
        return output

    def set_color_scale(self, worksheet, cell_range,
                        colors = ("#F8696B", "#FFEB84", "#63BE7B")):
        """Set Color Scale.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet that receives the conditional format.
        cell_range : str or list or tuple
            Cell range in A1 notation (``"B2:B6"``) or ``[first_row, first_col, last_row, last_col]`` (zero-based).
        colors : tuple of str, default ("#F8696B", "#FFEB84", "#63BE7B")
            Hex colors: three give a 3-color scale (minimum, midpoint, maximum), two give a 2-color scale (minimum, maximum).

        Returns
        -------
        int
            Always 0.

        Raises
        ------
        ValueError
            If ``colors`` does not hold exactly 2 or 3 colors.
        """
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
        """Set Data Bar in a Worksheet for a range of cells.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet that receives the conditional format.
        cell_range : str or list or tuple
            Cell range in A1 notation (``"B2:B6"``) or ``[first_row, first_col, last_row, last_col]`` (zero-based).
        bar_color : str, default "#63C384"
            Hex color of the bars (Excel 2010 style data bars).

        Returns
        -------
        int
            Always 0.
        """
        
        f = {'type': 'data_bar',
             'data_bar_2010': True,
             "bar_color": bar_color}

        if isinstance(cell_range, list) or isinstance(cell_range, tuple):
            cell_range = self.to_cell_range_text(*cell_range)

        worksheet.conditional_format(cell_range, f)
        return 0

    def set_cell_format(self, worksheet, cell_range, cformat, cell_condition = None):
        """Set format for a range of cell.

        The format is applied as an Excel conditional format (type ``no_errors``, or ``cell`` when ``cell_condition`` is
        given), not as a static cell format.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet that receives the format.
        cell_range : str or list or tuple
            Cell range in A1 notation (``"B2:B6"``) or ``[first_row, first_col, last_row, last_col]`` (zero-based).
        cformat : str or xlsxwriter.format.Format
            Name of a format in ``dict_cell_format`` (``KeyError`` for an unknown name) or an xlsxwriter ``Format`` object.
        cell_condition : tuple or None, default None
            ``(criteria, value)``: format only the cells that meet the condition. ``criteria`` is an xlsxwriter cell criteria
            string (``">"``, ``"<="``, ``"=="``, ``"between"``, ``"not between"``, ...; it is lower-cased) and ``value`` the
            comparison value, or a ``(minimum, maximum)`` pair for ``"between"`` and ``"not between"``. None formats all cells
            of the range that do not contain an error. A value that is not a tuple (for example a list) is silently ignored,
            as if it were None.

        Returns
        -------
        int
            Always 0.
        """

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
        """Conditionally Highlight Cells Row by Row.

        Tests every element of ``condition_list`` with ``val_input_condition`` and highlights, in the row that belongs to the
        element, the cells of the column(s) given for the true case or for the false case.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet that receives the formats.
        start_row : int
            Zero-based row of the first element of ``condition_list``. Element ``i`` is in row ``start_row + i``, that is Excel
            row ``start_row + i + 1``.
        condition_list : list
            Values to test, one per row.
        condition : tuple or str
            Condition as understood by ``val_input_condition``: ``(operator, value)`` with the operator ``>``, ``<``, ``=``,
            ``>=``, ``<=`` (or ``gt``, ``lt``, ``eq``, ``gte``, ``lte``) and a numeric value, or a string for an equality test.
        ifelse_col : tuple of str or None
            ``(column_if_true, column_if_false)``: column letters (``"B"``) or column letter ranges (``"B:D"``). A falsy entry
            (None or ``""``) means that nothing is highlighted in that case.
        cformat : str or xlsxwriter.format.Format, default "YELLOW_BG"
            Format of the highlighted cells (see ``set_cell_format``).

        Returns
        -------
        int
            Always 0.

        Notes
        -----
        An operator that ``val_input_condition`` does not know yields None, which counts as the false case.
        """
        
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
        """Remove Temp Images.

        Deletes the temporary box plot images (``.tmp_image_<digits>.png``) that ``write_boxplot`` leaves in the current working
        directory.

        Parameters
        ----------
        img_pattern : str, default r".tmp_image_[0-9]+.png"
            Regular expression searched (``re.search``) in the names of the files found under the current directory (``"./"``,
            searched recursively).

        Returns
        -------
        int or None
            0 when at least one matching file was found and removed, None when nothing matched.

        Notes
        -----
        Files are removed by bare file name, so only matches located directly in the current working directory can be deleted;
        a match inside a sub-directory makes ``os.remove`` raise ``FileNotFoundError``. ``close_workbook`` calls this method
        after saving the workbook.
        """
        tmp_imgs = list_files(location = "./", pattern = img_pattern)
        if len(tmp_imgs) > 0:
            for f in tmp_imgs:
                os.remove(f)
            return 0
        return 

    def close_workbook(self):
        '''Finish writing the contents of the workbook and close the file.

        Saves the workbook to ``filepath`` (until then the file is empty) and then calls ``remove_tmp_img`` to delete the
        temporary box plot images from the working directory.
        '''
        self.workbook.close()
        self.remove_tmp_img(img_pattern = r".tmp_image_[0-9]+.png")
        logging.info("All temp images have been removed.")


class ExcelMaster(ExcelWorkbook):
    """Write anything to Excel (Worksheet-level Operator)

    Keeps a zero-based cursor (``curr_row``, ``curr_col``). Every writing method starts at the cursor, or at ``loc`` when
    given, and then moves the cursor below (``skipby='row'``) or to the right of (``skipby='col'``) the block it wrote,
    leaving ``gap_number`` blank cells. Only that one coordinate is updated: with ``skipby='row'`` the column of the cursor
    keeps its value, even when ``loc`` pointed to another column. The class builds on ``ExcelWorkbook`` (conditional
    formats) and ``ExcelFormat`` (workbook creation and the named cell formats) and adds tables, merged titles, text,
    images and charts.

    Parameters
    ----------
    filepath : str
        Path of the output ``.xlsx`` file. The file is created when the object is built and filled by ``close_workbook``.
    verbose : bool
        Required (it has no default). If True, the writing methods log the cell range they wrote through
        ``logging.info``.
    gap_number : int, default 2
        Spacing between consecutive blocks. The attribute ``gap_number`` is initialised to this value plus 1, so the
        default leaves 3 blank rows (or columns); assigning the attribute afterwards (``em.gap_number = 1``) sets the
        number of blank rows or columns exactly.
    init_loc : tuple of int, default (0, 0)
        Initial cursor ``(row, col)``, zero-based. ``add_worksheet`` resets the cursor to (0, 0) unless it is called with
        ``reset_loc=False``, so ``init_loc`` only matters in that case.

    Attributes
    ----------
    curr_row : int
        Row of the cursor (zero-based).
    curr_col : int
        Column of the cursor (zero-based).
    gap_number : int
        Blank rows (or columns) left after a block: exact for tables and merged cells, approximate for charts and images,
        whose size is measured in cells of the default size.
    default_row_height : int or float
        Height in pixels of a row (20 at the default scale). It is used to convert chart and image sizes into cells and is
        overwritten by ``set_cell_size`` (hence by ``add_worksheet`` with ``cell_scale`` True or a tuple).
    default_col_width : int or float
        Width in pixels of a column (64 at the default scale), updated like ``default_row_height``.
    max_nrows : int
        Number of rows of a worksheet (1048576).
    max_ncols : int
        Number of columns of a worksheet (16384).
    """
    def __init__(self, filepath, verbose, gap_number = 2, init_loc = (0, 0)):
        """Create the workbook for ``filepath``, set ``verbose`` and initialise the cursor and the cell size defaults.

        The parameters and attributes are described in the class docstring.
        """
        super().__init__(filepath, verbose)
        self.curr_row = init_loc[0]
        self.curr_col = init_loc[1]
        self.gap_number = gap_number + 1

        self.default_row_height = 20
        self.default_col_width = 64

        self.max_nrows = 1048576
        self.max_ncols = 16384

    def add_worksheet(self, name, hide_grid = True, reset_loc = True, cell_scale = True, auto_fit = False, zoom_perc = 100, tab_color = None):
        """Add a worksheet.

        Parameters
        ----------
        name : str
            Name of the worksheet (unique in the workbook, at most 31 characters).
        hide_grid : bool, default True
            If True, hide the gridlines on screen and in print.
        reset_loc : bool, default True
            If True, reset the cursor to (0, 0).
        cell_scale : bool or tuple of float, default True
            ``True`` writes an explicit size for every row and column of the sheet (20 px high and 64 px wide), which takes a
            few seconds and adds a few MB to the file; a tuple ``(row_scale, col_scale)`` multiplies that size (for example
            ``(1, 2)`` makes the columns twice as wide). Any other value (``False``, None) leaves the sizes of the sheet and
            ``default_row_height`` / ``default_col_width`` untouched.
        auto_fit : bool, default False
            Meant to fit the column widths automatically. ``True`` calls ``worksheet.auto_fit()``, which xlsxwriter does not
            provide (it has ``autofit()``), so it raises ``AttributeError``; keep it False.
        zoom_perc : int, default 100
            Zoom of the worksheet in percent (xlsxwriter accepts 10 to 400).
        tab_color : str or None, default None
            Color of the sheet tab, a hex code such as ``"#FF0000"`` or a color name such as ``"red"``. None keeps the default.

        Returns
        -------
        xlsxwriter.worksheet.Worksheet
            The new worksheet, also registered in ``ws_dict`` and in the pandas writer so that DataFrames can be written to it.

        Notes
        -----
        ``cell_scale`` True or a tuple overwrites ``default_row_height`` and ``default_col_width``, which later charts and
        images use to convert their size into cells.
        """
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
        """Reset Current Location.

        Parameters
        ----------
        loc : tuple of int, default (0, 0)
            New cursor position ``(row, col)``, zero-based.

        Returns
        -------
        int
            Always 0.
        """
        
        self.curr_row = loc[0]
        self.curr_col = loc[1]
        return 0

    def _reset_cell_size(self):
        """ Reset Cell Size to Default Value. """
        self.default_row_height = 20
        self.default_col_width = 64
        return 0

    def set_cell_size(self, worksheet, size_scale = (1, 1)):
        """Set Cell Size in Scale.

        Sets the pixel height of every row (all 1,048,576) and the pixel width of every column of the worksheet, and stores the
        sizes in ``default_row_height`` and ``default_col_width``, which later chart and image sizes are converted with.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet to resize.
        size_scale : tuple of float, default (1, 1)
            ``(row_scale, col_scale)`` multiplying the base cell size of 20 px high by 64 px wide. A value that is not a
            2-tuple leaves ``default_row_height`` and ``default_col_width`` as they are, and these are applied again.

        Returns
        -------
        int
            Always 0.

        Notes
        -----
        Writes the size of every row, which takes a few seconds and adds a few MB to the file.
        """
        
        if isinstance(size_scale, tuple) and len(size_scale) == 2:
            self._reset_cell_size()
            self.default_row_height = self.default_row_height * size_scale[0]
            self.default_col_width =  self.default_col_width * size_scale[1]

        for i in range(0, self.max_nrows):
            worksheet.set_row_pixels(i, height=self.default_row_height)
            
        worksheet.set_column_pixels(0, self.max_ncols - 1, width=self.default_col_width)
        return 0

    def get_curr_loc(self, toCell = False):
        """Get Current Location in worksheet.

        Parameters
        ----------
        toCell : bool, default False
            If True, return the cursor as an A1-style cell reference instead of a tuple.

        Returns
        -------
        tuple of int or str
            ``(curr_row, curr_col)`` (zero-based), or for example ``"C4"`` when ``toCell`` is True.
        """
        if toCell:
            return xl_rowcol_to_cell(self.curr_row, self.curr_col)
        return (self.curr_row, self.curr_col)
    
    def set_border_line(self, worksheet, valuerange, border_line = 1):
        """Set border line for a range of cells.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet that receives the border.
        valuerange : str or list or tuple
            Cell range in A1 notation (``"B2:D6"``) or ``[first_row, first_col, last_row, last_col]`` (zero-based).
        border_line : int, default 1
            xlsxwriter border style index, used for the top, bottom, left and right border of every cell of the range
            (1 thin, 2 medium, 5 thick, ...).

        Returns
        -------
        int
            Always 0.

        Notes
        -----
        The border is applied as a conditional format (through ``set_cell_format``).
        """
        border_fmt = self.workbook.add_format({'bottom': border_line, 'top': border_line, 'left': border_line, 'right': border_line})
        self.set_cell_format(worksheet = worksheet, cell_range = valuerange, cformat = border_fmt)
        return 0

    def merge_col(self, worksheet, loc = None, nrows = 1, ncols = 1, text = "", skipby = 'row', cformat = 'BLUE_H4', retCellRange = None):
        """Merge columns in a single row.

        Merges the block of ``nrows`` x ``ncols`` cells that starts at ``loc`` (one row by default) and writes ``text`` in it.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet to write on.
        loc : tuple of int or None, default None
            ``(row, col)`` (zero-based) of the top-left cell. None starts at the cursor.
        nrows : int, default 1
            Number of rows to merge.
        ncols : int, default 1
            Number of columns to merge.
        text : str, default ""
            Text written in the merged cell.
        skipby : str or None, default 'row'
            Cursor move after the write: ``'row'`` sets ``curr_row`` to ``start_row + nrows + gap_number``, ``'col'`` sets
            ``curr_col`` to ``start_col + ncols + gap_number``; any other value (for example None) leaves the cursor unchanged.
        cformat : str, default 'BLUE_H4'
            Name of a format in ``dict_cell_format``. An unknown name, and also an xlsxwriter ``Format`` object, raises
            ``KeyError``.
        retCellRange : str or None, default None
            ``"text"`` returns the written range in A1 notation (for example ``"A1:C1"``); ``"value"`` returns
            ``[first_row, first_col, last_row, last_col]`` (zero-based); None returns 0.

        Returns
        -------
        int or str or list of int
            0, or the written range as selected by ``retCellRange``.

        Notes
        -----
        A block of one row and one column is written as a plain cell, because xlsxwriter cannot merge a single cell.
        """
        
        start_row = loc[0] if loc else self.curr_row
        start_col = loc[1] if loc else self.curr_col
        written_range = [start_row, start_col, start_row + nrows - 1, start_col + ncols - 1]
        
        if nrows == 1 and ncols == 1:
            # xlsxwriter refuses to merge a single cell and writes nothing, so a
            # one-column table used to lose its title; write the cell directly.
            worksheet.write(start_row, start_col, text, self.dict_cell_format[cformat])
        else:
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
        """Write a dataframe to excel file.

        Writes the optional title row (merged across the table), the header and the values with ``DataFrame.to_excel``, then
        formats the header and the value cells.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet to write on. The table goes through the pandas writer, which finds the sheet by its name.
        df : pandas.DataFrame
            Table to write.
        loc : tuple of int or None, default None
            ``(row, col)`` (zero-based) of the top-left cell, the title row if there is one. None starts at the cursor.
        title : str or None, default None
            Title written in a merged row above the table (format ``titleformat``). None or an empty string writes no title
            row.
        index : bool, default False
            If True, write the index as the leading column(s), one per index level.
        header : bool, default True
            If True, write the column names, one header row per column level.
        skipby : str or None, default 'row'
            Cursor move after the write: ``'row'`` sets ``curr_row`` to ``start_row + nrows + gap_number`` (``nrows`` counts
            the title and header rows), ``'col'`` sets ``curr_col`` to ``start_col + ncols + gap_number`` (``ncols`` counts
            the index columns); any other value (for example None) leaves the cursor unchanged.
        titleformat : str, default "BLUE_H4"
            Name of a format in ``dict_cell_format`` for the title (a name only: a ``Format`` object raises ``KeyError``).
        headerformat : str or xlsxwriter.format.Format, default "TABLE_HEADER"
            Format of the header cells: a name in ``dict_cell_format`` or a ``Format`` object.
        valueformat : str or xlsxwriter.format.Format, default "----"
            Format of every value cell, as a name in ``dict_cell_format`` or a ``Format`` object. It applies to all columns, so
            a percentage format would also turn count columns into percentages; format single columns afterwards with
            ``set_cell_format``.
        retCellRange : str or None, default None
            ``"text"`` returns the written range in A1 notation (for example ``"A1:D6"``); ``"value"`` returns
            ``[first_row, first_col, last_row, last_col]`` (zero-based), title and header included; None returns 0.

        Returns
        -------
        int or str or list of int
            0, or the written range as selected by ``retCellRange``.

        Notes
        -----
        The header and value formats are applied as conditional formats (see ``set_cell_format``).
        """
        
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
        """Insert an image to the sheet.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet to write on.
        figPath : str
            Path of the image file (for example PNG or JPEG).
        figScale : tuple of float, default (1, 1)
            ``(x_scale, y_scale)`` factors for the width and the height of the image: 1 keeps the size, 0.8 shrinks it by 20
            percent. They are factors, not pixels.
        loc : tuple of int or None, default None
            ``(row, col)`` (zero-based) of the cell that holds the top-left corner of the image. None starts at the cursor.
        skipby : str or None, default 'row'
            Cursor move after the write: ``'row'`` moves the cursor below the image, ``'col'`` to the right of it, by the size
            of the image in cells plus ``gap_number``; any other value (for example None) leaves the cursor unchanged.
        retCellRange : str or None, default None
            ``"text"`` returns the covered range in A1 notation; ``"value"`` returns
            ``[first_row, first_col, last_row, last_col]`` (zero-based); None returns 0.

        Returns
        -------
        int or str or list of int
            0, or the covered range as selected by ``retCellRange``.

        Notes
        -----
        The size of the image in cells is computed from the scaled pixel size and ``default_row_height`` /
        ``default_col_width`` (rows rounded down, columns rounded up), so the returned range and the cursor move are
        approximate.
        """
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
        """Write line chart to Excel worksheet.

        Builds an xlsxwriter chart from ``df`` and inserts it at ``loc`` (or at the cursor). Despite the summary, ``chart_type``
        also gives column, stacked column and pie charts. The source data of the chart are written to a hidden worksheet named
        ``__CHRT_DATA_<N>``, or to ``worksheet`` itself with ``outputData=True``.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet that receives the chart.
        df : pandas.DataFrame
            Source data, one row per category.
        y_list : list of str
            Columns of ``df`` to plot: one series per column, named after the column.
        x : str or list of str or None, default None
            Column(s) of ``df`` that give the category labels (several columns give multi-level labels). None uses the index
            of ``df`` (all its levels).
        title : str, default ""
            Title of the chart.
        chart_size : tuple of int, default (30, 13)
            ``(rows, columns)`` measured in worksheet cells, not pixels: the chart is ``chart_size[0] * default_row_height``
            pixels high and ``chart_size[1] * default_col_width`` pixels wide.
        chart_type : str or dict, default "line"
            ``"line"``, ``"column"``, ``"stacked_column"`` or ``"pie"``; or an options dict for ``workbook.add_chart`` such as
            ``{"type": "line"}``.
        y_axis_range : tuple, default (None, None)
            ``(min, max)`` of the value axis; None entries (or None instead of the tuple) mean automatic scaling.
        y_num_format : str or None, default None
            Number format of the value axis, for example ``"0.00%"``. None sets no number format.
        y2_axis : bool, default False
            If True, plot the series on the secondary y axis; the axis settings (range, format, title, gridlines) then apply
            to that axis.
        loc : tuple of int or None, default None
            ``(row, col)`` (zero-based) of the top-left cell of the block. None starts at the cursor.
        retChart : bool, default False
            If True, return the chart object instead of inserting it: nothing is inserted and the cursor does not move. Use it
            with ``write_combined_chart`` or ``append_to_chart``.
        skipby : str or None, default "row"
            Cursor move after the insertion: ``'row'`` moves the cursor below the chart, ``'col'`` to the right of it, by the
            size of the chart (and of the data table with ``outputData``) plus ``gap_number``; any other value (for example
            None) leaves the cursor unchanged.
        outputData : bool, default False
            If True, write the table of chart data to ``worksheet`` at the insertion point and put the chart below it; the
            hidden data sheet that is created anyway then stays empty.
        retCellRange : str or None, default None
            ``"text"`` returns the covered range in A1 notation; ``"value"`` returns
            ``[first_row, first_col, last_row, last_col]`` (zero-based); None returns 0. The range is approximate.
        xy_axes_name : tuple of str, default ("", "")
            ``(x axis title, y axis title)``.
        major_gridlines : bool, default False
            If True, show the major gridlines of the value axis.
        legend : str, default "bottom"
            xlsxwriter legend position: ``"top"``, ``"bottom"``, ``"left"``, ``"right"``, ``"overlay_left"``,
            ``"overlay_right"``, or ``"none"`` to hide the legend.
        line_marker : str, default "circle"
            Marker type of every series (xlsxwriter ``marker`` type, for example ``"circle"``, ``"square"``, ``"none"``).
        line_type : str, default "solid"
            Dash type of the line of every series (xlsxwriter ``dash_type``, for example ``"solid"``, ``"dash"``,
            ``"long_dash"``).
        chart_style : int or None, default None
            xlsxwriter built-in chart style number (``chart.set_style``). None keeps the default style.
        append_to_chart : xlsxwriter.chart.Chart or None, default None
            Existing chart (obtained with ``retChart=True``) that receives the series instead of a new chart. It must be of the
            type given by ``chart_type`` (line, column or stacked column, pie); otherwise a warning is issued and a new chart
            is used.

        Returns
        -------
        xlsxwriter.chart.Chart or int or str or list of int
            The chart when ``retChart`` is True; otherwise 0, or the range covered by the chart as selected by
            ``retCellRange``.

        Raises
        ------
        ValueError
            If ``chart_type`` is a string other than ``"line"``, ``"column"``, ``"stacked_column"`` and ``"pie"``.

        Notes
        -----
        Every call adds a hidden worksheet ``__CHRT_DATA_<N>`` to the workbook (it is registered in ``ws_dict``).
        """

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
        """Combined two chart objects and then write to Excel worksheet.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet that receives the combined chart.
        chart1 : xlsxwriter.chart.Chart
            Primary chart, obtained with ``retChart=True``. ``chart2`` is combined into it and it is the object inserted, so
            its title, size and primary axis apply.
        chart2 : xlsxwriter.chart.Chart
            Secondary chart, obtained with ``retChart=True``, combined into ``chart1`` (xlsxwriter supports only some
            combinations, for example a column chart with a line chart).
        loc : tuple of int or None, default None
            ``(row, col)`` (zero-based) of the top-left cell. None starts at the cursor.
        chart_size : tuple of int, default (30, 13)
            ``(rows, columns)`` in cells. It is used only to compute the returned range and the cursor move; it does not
            resize the chart, whose size was set when the charts were created.
        skipby : str or None, default "row"
            Cursor move after the insertion: ``'row'`` moves the cursor below the chart, ``'col'`` to the right of it, by
            ``chart_size`` plus ``gap_number``; any other value (for example None) leaves the cursor unchanged.
        retCellRange : str or None, default None
            ``"text"`` returns the covered range in A1 notation; ``"value"`` returns
            ``[first_row, first_col, last_row, last_col]`` (zero-based); None returns 0.

        Returns
        -------
        int or str or list of int
            0, or the covered range as selected by ``retCellRange``.
        """
        
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
        """Write duo-chart sharing the same x axis.

        Builds two charts from ``df`` and ``x`` with ``write_chart`` (the first from ``y1_list`` on the primary y axis, the
        second from ``y2_list`` on the secondary y axis), combines them and inserts the result. Typical use: columns for counts
        on the left axis and a line for a rate on the right axis.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet that receives the chart.
        df : pandas.DataFrame
            Source data, one row per category.
        y1_list : list of str
            Columns drawn on the primary y axis, in the chart type ``c1_type``.
        y2_list : list of str, default None
            Columns drawn on the secondary y axis, in the chart type ``c2_type``. In practice it is required: the default None
            makes the second chart fail (``KeyError`` or ``TypeError``).
        x : str or list of str or None, default None
            Column(s) that give the category labels, as in ``write_chart``. None uses the index of ``df``.
        c1_type : str or dict, default "column"
            Type of the first chart: ``"line"``, ``"column"``, ``"stacked_column"``, ``"pie"`` or an options dict for
            ``workbook.add_chart``.
        c2_type : str or dict, default "line"
            Type of the second chart, as ``c1_type``.
        y1_axis_range : tuple or None, default (0, 1)
            ``(min, max)`` of the primary y axis. The default fixes the axis to 0 - 1, which suits rates but clips counts; pass
            None (or ``(None, None)``) for automatic scaling.
        y2_axis_range : tuple or None, default None
            ``(min, max)`` of the secondary y axis. None uses ``y1_axis_range``.
        y1_num_format : str or None, default None
            Number format of the primary y axis, for example ``"0.00%"``.
        y2_num_format : str or None, default None
            Number format of the secondary y axis.
        y1_line_marker : str, default "circle"
            Marker type of the series of the first chart.
        y2_line_marker : str, default "circle"
            Marker type of the series of the second chart.
        y1_line_type : str, default "solid"
            Dash type of the lines of the first chart.
        y2_line_type : str, default "solid"
            Dash type of the lines of the second chart.
        loc : tuple of int or None, default None
            ``(row, col)`` (zero-based) of the top-left cell. None starts at the cursor.
        title : str, default ""
            Title of the chart (the combined chart shows the title of the first one).
        chart_size : tuple of int, default (30, 13)
            ``(rows, columns)`` measured in worksheet cells, not pixels.
        xy_axes_name : tuple of str, default ("", "")
            ``(x axis title, primary y title)`` or ``(x axis title, primary y title, secondary y title)``; with two entries the
            secondary y axis has no title.
        major_gridlines : bool, default False
            If True, show the major gridlines of the primary y axis; the secondary axis never shows gridlines.
        retChart : bool, default False
            If True, return the tuple ``(chart1, chart2)`` instead of inserting anything (the cursor does not move).
        retCellRange : str or None, default None
            ``"text"`` returns the covered range in A1 notation; ``"value"`` returns
            ``[first_row, first_col, last_row, last_col]`` (zero-based); None returns 0.
        skipby : str or None, default "row"
            Cursor move after the insertion: ``'row'`` moves the cursor below the chart, ``'col'`` to the right of it; any other
            value (for example None) leaves the cursor unchanged.

        Returns
        -------
        tuple of xlsxwriter.chart.Chart or int or str or list of int
            The tuple ``(chart1, chart2)`` when ``retChart`` is True; otherwise the result of ``write_combined_chart``: 0, or the
            covered range as selected by ``retCellRange``.

        Notes
        -----
        Two hidden worksheets ``__CHRT_DATA_<N>`` (one per chart) are added to the workbook.
        """

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
        """Write text using Python dictionary.

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet to write on.
        dict_cells : dict
            Maps a cell (``"A1"``) or a range to merge (``"M1:O2"``) to its content; formats are names in
            ``dict_cell_format``:

            * range key with ``[text, format_name]``: merge the range and write ``text`` in it;
            * cell key with ``[text, format_name]``: write ``text`` with the format;
            * cell key with ``[[item, item, ...]]`` (a one-element list that holds a list): write a rich string. An item that
              starts with ``"~~~"`` is a format name (``"~~~B"``) that applies to the text items that follow; the other items
              are text.

        Returns
        -------
        None
            Nothing is returned. The cursor is not used or changed.
        """
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
        r"""Write text content line by line to the worksheet.

        Every line becomes one row, starting at ``loc`` (or the cursor), in a single column. Inline markup in a line:

        * ``{FORMAT_NAME}`` styles the text that follows it with a format of ``dict_cell_format`` (``{B}``, ``{I}``,
          ``{HEADER_2}``, ``{BLUE_H2}``, ...). Several signs in one line give a rich string with mixed formats (text before
          the first sign is allowed then). A line with exactly one sign must start with it, otherwise ``TypeError``.
        * ``" [>] "`` (with the blanks) splits a line into cells that are written side by side in the same row.
        * a trailing ``[[FORMAT_NAME]]`` formats the whole cell (as a conditional format).

        Parameters
        ----------
        worksheet : xlsxwriter.worksheet.Worksheet
            Worksheet to write on.
        input_text : str or None, default None
            Text to write, lines separated by ``"\n"``.
        txt_path : str or None, default None
            Path of a text file whose lines are written. Give exactly one of ``input_text`` and ``txt_path``.
        loc : tuple of int or None, default None
            ``(row, col)`` (zero-based) of the first line. None starts at the cursor.
        retCellRange : str or None, default None
            ``"text"`` returns the range in A1 notation; ``"value"`` returns ``[first_row, first_col, last_row, last_col]``
            (zero-based); None returns 0. The range covers only the first column, and its last row is one past the last line.

        Returns
        -------
        int or str or list of int
            0, or the range as selected by ``retCellRange``.

        Raises
        ------
        ValueError
            If both or neither of ``input_text`` and ``txt_path`` are given.

        Notes
        -----
        The cursor is set to the last row written (``curr_row = start_row + number_of_lines - 1``) and ``curr_col`` is not
        changed, even when ``loc`` is given. End ``input_text`` with ``"\n"`` so that the next block starts below the text;
        otherwise it overwrites the last line. The file given by ``txt_path`` is opened and not closed explicitly.
        """
        
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
        """Plot Box Plot Chart using matplotlib.

        Draws one box per distinct value of ``x`` (in ascending order) from the values of ``y``. Both axis labels are
        reformatted with ``string_proc`` (``"credit_score"`` becomes ``"Credit Score"``). The figure is closed before the
        function returns.

        Parameters
        ----------
        df : pandas.DataFrame
            Source data.
        x : str
            Column that defines the groups, one box per distinct value.
        y : str
            Numeric column whose distribution is drawn in each box.
        y_percentage : bool, default False
            If True, format the y axis as percent (``PercentFormatter(100)``).
        colored_box : bool, default True
            If True, fill the boxes with the colors of ``color_grp`` (alpha 0.7); otherwise draw unfilled boxes.
        color_grp : tuple or str or list of str, default (10, 1)
            Box colors, as accepted by ``color_input_validation``: a ``(start_num, step)`` tuple that picks XKCD named colors, one
            color code for all boxes, or a list with one color code per box. Used only when ``colored_box`` is True.
        title : str, default ""
            Title of the plot.
        fontsize : int, default 14
            Font size of the axis labels and ticks (the title is ``fontsize + 4``).
        figsize : tuple of float, default (8, 6)
            Figure size ``(width, height)`` in inches (the figure is created with 200 dpi).
        show_fig : bool, default True
            If True, display the figure with ``plt.show()``.
        img_path : str or None, default None
            If given, save the figure to this path (200 dpi).
        transp_bg : bool, default False
            If True, save the image with a transparent background.

        Returns
        -------
        tuple
            ``(fig, ax)``: the matplotlib ``Figure`` and ``Axes``.

        Notes
        -----
        The values of ``y`` are always multiplied by 100: the call that prepares the data passes ``y_percentage=True`` whatever
        this argument is, so ``y_percentage=False`` only skips the percent formatting of the axis (which then shows values 100
        times larger than ``y``). It is meant for rates between 0 and 1.
        """
        
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
        """Write boxplot to Worksheet.

        Draws the box plot with ``plot_boxplot`` into a temporary PNG file (``./.tmp_image_<date time><random number>.png`` in
        the working directory), resizes the file to ``figsize`` worksheet cells and inserts it with ``insert_image``.

        Parameters
        ----------
        ws : xlsxwriter.worksheet.Worksheet
            Worksheet to write on (note the name: ``ws``, not ``worksheet``).
        df : pandas.DataFrame
            Source data.
        x : str
            Column that defines the groups, one box per distinct value.
        y : str
            Numeric column whose distribution is drawn in each box (always multiplied by 100, see ``plot_boxplot``).
        y_percentage : bool, default False
            If True, format the y axis as percent.
        colored_box : bool, default True
            If True, fill the boxes with the colors of ``color_grp``.
        color_grp : tuple or str or list of str, default (10, 1)
            Box colors, as accepted by ``color_input_validation``.
        title : str, default ""
            Title of the plot.
        fontsize : int, default 14
            Font size of the axis labels and ticks.
        figsize : tuple of int, default (30, 13)
            Size of the inserted image as ``(rows, columns)`` of worksheet cells, not inches (unlike ``plot_boxplot``): the
            image file is resized to ``columns * default_col_width`` by ``rows * default_row_height`` pixels. It is not passed to
            the figure itself.
        show_fig : bool, default False
            If True, display the figure with ``plt.show()``.
        img_path : str or None, default None
            Not used: the image is always saved to the temporary file.
        transp_bg : bool, default False
            If True, save the image with a transparent background.
        loc : tuple of int or None, default None
            ``(row, col)`` (zero-based) of the top-left cell of the image. None starts at the cursor.
        skipby : str or None, default "row"
            Cursor move after the insertion, as in ``insert_image``.
        retCellRange : str or None, default None
            ``"text"`` returns the covered range in A1 notation; ``"value"`` returns
            ``[first_row, first_col, last_row, last_col]`` (zero-based); None returns 0.

        Returns
        -------
        int or str or list of int
            The result of ``insert_image``: 0, or the covered range as selected by ``retCellRange``.

        Notes
        -----
        The temporary image stays in the working directory until ``close_workbook`` (or ``remove_tmp_img``) deletes it.
        """
        
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