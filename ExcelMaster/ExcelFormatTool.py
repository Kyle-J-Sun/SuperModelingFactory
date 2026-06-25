import xlsxwriter
import pandas as pd
from datetime import datetime
import pdb, os
import logging
logger = logging.getLogger(__name__)

class ExcelFormat:
    '''
    Initialize excel format.
    
    '''
    
    def __init__(self, filepath):
        '''
        This creates the workbook using the filename specified in
        filename_workbook.
        
        '''

        self.engine = pd.ExcelWriter(filepath, engine='xlsxwriter')
        self.workbook = self.engine.book
        self.basename = os.path.basename(filepath)
        self.base_filepath = filepath.strip(self.basename)


        ######### Basic Text-related Formatting ###############
        
        self.dict_cell_format = {}
        
        self.TEXT_NO_FORMAT = self.workbook.add_format({'font_name': 'Calibri', 'font_size': 11})
        self.dict_cell_format['TEXT_NO_FORMAT'] = self.TEXT_NO_FORMAT
        self.dict_cell_format[''] = self.TEXT_NO_FORMAT

        self.HEADER_1 = self.workbook.add_format({
            'bold': True,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 18,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['HEADER_1'] = self.HEADER_1
        self.dict_cell_format['#'] = self.HEADER_1

        self.HEADER_2 = self.workbook.add_format({
            'bold': True,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 16,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['HEADER_2'] = self.HEADER_2
        self.dict_cell_format['##'] = self.HEADER_2

        self.HEADER_3 = self.workbook.add_format({
            'bold': True,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 14,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['HEADER_3'] = self.HEADER_3
        self.dict_cell_format['###'] = self.HEADER_3

        self.HEADER_4 = self.workbook.add_format({
            'bold': True,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 12,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['HEADER_4'] = self.HEADER_4
        self.dict_cell_format['####'] = self.HEADER_4

        self.HEADER_1I = self.workbook.add_format({
            'bold': True,
            'underline': True,            
            'font_name': 'Calibri',
            'font_size': 18,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['HEADER_1I'] = self.HEADER_1I
        self.dict_cell_format['#_'] = self.HEADER_1I

        self.HEADER_2I = self.workbook.add_format({
            'bold': True,
            'underline': True,            
            'font_name': 'Calibri',
            'font_size': 16,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['HEADER_2I'] = self.HEADER_2I
        self.dict_cell_format['##_'] = self.HEADER_2I

        self.HEADER_3I = self.workbook.add_format({
            'bold': True,
            'underline': True,            
            'font_name': 'Calibri',
            'font_size': 14,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['HEADER_3I'] = self.HEADER_3I
        self.dict_cell_format['###_'] = self.HEADER_3I

        self.HEADER_4I = self.workbook.add_format({
            'bold': True,
            'underline': True,            
            'font_name': 'Calibri',
            'font_size': 12,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['HEADER_4I'] = self.HEADER_4I
        self.dict_cell_format['####_'] = self.HEADER_4I

        self.TEXT_BOLD = self.workbook.add_format({
            'bold': True,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#000000',
            'align': 'left'
        })
        self.dict_cell_format['TEXT_BOLD'] = self.TEXT_BOLD
        self.dict_cell_format['B'] = self.TEXT_BOLD
        self.dict_cell_format['**'] = self.TEXT_BOLD

        self.TEXT_ITALIC = self.workbook.add_format({
            'bold': False,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#000000',
            'align': 'left',
            'italic': True
        })
        self.dict_cell_format['TEXT_ITALIC'] = self.TEXT_ITALIC
        self.dict_cell_format['I'] = self.TEXT_ITALIC
        self.dict_cell_format['*'] = self.TEXT_ITALIC

        self.TEXT_UNDERLINE = self.workbook.add_format({
            'bold': False,
            'underline': True,            
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#000000',
            'align': 'left',
            'italic': False
        })
        self.dict_cell_format['TEXT_UNDERLINE'] = self.TEXT_UNDERLINE
        self.dict_cell_format['U'] = self.TEXT_UNDERLINE
        self.dict_cell_format['_'] = self.TEXT_UNDERLINE

        self.TEXT_NO_FORMAT_BORDER = self.workbook.add_format({
            'bold': False,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#000000',
            'align': 'left',
            'border': 1
        })
        self.dict_cell_format['TEXT_NO_FORMAT_BORDER'] = self.TEXT_NO_FORMAT_BORDER  
        self.dict_cell_format['BORDER'] = self.TEXT_NO_FORMAT_BORDER
        self.dict_cell_format['----'] = self.TEXT_NO_FORMAT_BORDER
        
        self.TEXT_BORDER_CENTER = self.workbook.add_format({
            'bold': False,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#000000',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        self.dict_cell_format['TEXT_BORDER_CENTER'] = self.TEXT_BORDER_CENTER  
        self.dict_cell_format['BORDER_CENTER'] = self.TEXT_BORDER_CENTER
        self.dict_cell_format['----C'] = self.TEXT_BORDER_CENTER

        self.TEXT_RED = self.workbook.add_format({
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#FF3800',
            'align': 'left'
        })
        self.dict_cell_format['TEXT_RED'] = self.TEXT_RED
        self.dict_cell_format['RED'] = self.TEXT_RED

        self.TEXT_RED_BOLD = self.workbook.add_format({
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#FF3800',
            "bold": True,
            'align': 'left'
        })
        self.dict_cell_format['TEXT_RED_BOLD'] = self.TEXT_RED_BOLD
        self.dict_cell_format['**RED'] = self.TEXT_RED_BOLD

        self.TEXT_BOLD_UNDERLINE = self.workbook.add_format({
            'bold': True,
            'underline': True,
            'font_name': 'Calibri',
            'font_size': 11,
            'align': 'left'
        })
        self.dict_cell_format['TEXT_BOLD_UNDERLINE'] = self.TEXT_BOLD_UNDERLINE
        self.dict_cell_format['BU'] = self.TEXT_BOLD_UNDERLINE
        self.dict_cell_format['**_'] = self.TEXT_BOLD_UNDERLINE

        self.TEXT_ITALIC_UNDERLINE = self.workbook.add_format({
            'bold': False,
            'underline': True,
            'italic': True,
            'font_name': 'Calibri',
            'font_size': 11,
            'align': 'left'
        })
        self.dict_cell_format['TEXT_ITALIC_UNDERLINE'] = self.TEXT_ITALIC_UNDERLINE
        self.dict_cell_format['IU'] = self.TEXT_ITALIC_UNDERLINE
        self.dict_cell_format['*_'] = self.TEXT_ITALIC_UNDERLINE

        self.TEXT_BOLD_ITALIC_UNDERLINE = self.workbook.add_format({
            'bold': True,
            'underline': True,
            'italic': True,
            'font_name': 'Calibri',
            'font_size': 11,
            'align': 'left'
        })
        self.dict_cell_format['TEXT_BOLD_ITALIC_UNDERLINE'] = self.TEXT_BOLD_ITALIC_UNDERLINE
        self.dict_cell_format['BIU'] = self.TEXT_BOLD_ITALIC_UNDERLINE
        self.dict_cell_format['***_'] = self.TEXT_BOLD_ITALIC_UNDERLINE
        
        self.HEADER_3_BOLD_UNDERLINE = self.workbook.add_format({
            'bold': True,
            'underline': True,
            'font_name': 'Calibri',
            'font_size': 14,
            'align': 'left'
        })
        self.dict_cell_format['HEADER_4_BOLD_UNDERLINE'] = self.HEADER_3_BOLD_UNDERLINE
        self.dict_cell_format['H3_BU'] = self.HEADER_3_BOLD_UNDERLINE
        self.dict_cell_format['####**_'] = self.HEADER_3_BOLD_UNDERLINE

        self.HEADER_4_BOLD_UNDERLINE = self.workbook.add_format({
            'bold': True,
            'underline': True,
            'font_name': 'Calibri',
            'font_size': 12,
            'align': 'left'
        })
        self.dict_cell_format['HEADER_4_BOLD_UNDERLINE'] = self.HEADER_4_BOLD_UNDERLINE
        self.dict_cell_format['H4_BU'] = self.HEADER_4_BOLD_UNDERLINE
        self.dict_cell_format['####**_'] = self.HEADER_4_BOLD_UNDERLINE

        self.TEXT_NO_FORMAT_PERCENTAGE = self.workbook.add_format({
            'bold': False,
            'underline': False,            
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#000000',
            'num_format': '0.0000%',
            'align': 'left',    
            'border': 1
        })
        self.dict_cell_format['TEXT_NO_FORMAT_PERCENTAGE'] = self.TEXT_NO_FORMAT_PERCENTAGE
        self.dict_cell_format['NUM_PERCENTAGE'] = self.TEXT_NO_FORMAT_PERCENTAGE
        self.dict_cell_format['%'] = self.TEXT_NO_FORMAT_PERCENTAGE

        self.TEXT_SECTION_HEADER = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 12,
            'font_color': '#1F497D',
            'bg_color': '#C5D9F1',
            'align': 'center',
            'valign': 'vcenter',
            'text_wrap': False
        })
        self.dict_cell_format['TEXT_SECTION_HEADER'] = self.TEXT_SECTION_HEADER
        self.dict_cell_format['SECTION'] = self.TEXT_SECTION_HEADER

        self.TEXT_ITALIC_10 = self.workbook.add_format({
            'italic': True,
            'font_name': 'Calibri',
            'font_size': 10,
            'align': 'left'
        })
        self.dict_cell_format['TEXT_ITALIC_10'] = self.TEXT_ITALIC_10

        ######### Basic Text-related Formatting (END) ###############


        ######### TABLE HEADER FORMATTING  ##########################

        self.TABLE_ROW_COL_HEADER = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#1F4994',
            'align': 'left',
            'border': 1
        })
        self.dict_cell_format['ROW_COL_HEADER'] = self.TABLE_ROW_COL_HEADER

        self.TABLE_HEADER = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#000000',
            # No 'bg_color' key = no fill. xlsxwriter >= 3.2 rejects None via
            # Color._from_value(None) -> TypeError, so omit the key entirely.
            'align': 'center',
            'valign': 'vcenter',
            'text_wrap': False,
            'border': 1
        })
        self.dict_cell_format['TABLE_HEADER'] = self.TABLE_HEADER
        self.dict_cell_format['HEADER'] = self.TABLE_HEADER
        
        ######### TABLE HEADER FORMATTING (END)  ##########################

        ######## Colored Title Formatting #################################
        
        self.BG_BOLD_ORANGE_H4 = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 12,
            'font_color': '#000000',
            'bg_color': '#FABF8F',
            'align': 'center',
            'border': 1
        })
        self.dict_cell_format['BG_BOLD_ORANGE_H4'] = self.BG_BOLD_ORANGE_H4
        self.dict_cell_format['ORANGE_H4'] = self.BG_BOLD_ORANGE_H4
        
        self.BG_BOLD_ORANGE_H3 = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 14,
            'font_color': '#000000',
            'bg_color': '#FABF8F',
            'align': 'center',
            'border': 1
        })
        self.dict_cell_format['BG_BOLD_ORANGE_H3'] = self.BG_BOLD_ORANGE_H3
        self.dict_cell_format['ORANGE_H3'] = self.BG_BOLD_ORANGE_H3

        self.BG_BOLD_ORANGE_H2 = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 16,
            'font_color': '#000000',
            'bg_color': '#FABF8F',
            'align': 'center',
            'border': 1
        })
        self.dict_cell_format['BG_BOLD_ORANGE_H2'] = self.BG_BOLD_ORANGE_H2
        self.dict_cell_format['ORANGE_H2'] = self.BG_BOLD_ORANGE_H2
        
        self.BG_BOLD_ORANGE_H1 = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 18,
            'font_color': '#000000',
            'bg_color': '#FABF8F',
            'align': 'center',
            'border': 1
        })
        self.dict_cell_format['BG_BOLD_ORANGE_H1'] = self.BG_BOLD_ORANGE_H1
        self.dict_cell_format['ORANGE_H1'] = self.BG_BOLD_ORANGE_H1
        
        self.BG_FONT_BLUE_H4 = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 12,
            'font_color': '#1F497D',
            'bg_color': '#C5D9F1',
            'align': 'center'  ,
            'border': 1
        })
        self.dict_cell_format['BG_FONT_BLUE_H4'] = self.BG_FONT_BLUE_H4
        self.dict_cell_format['BLUE_H4'] = self.BG_FONT_BLUE_H4

        self.BG_FONT_BLUE_H3 = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 14,
            'font_color': '#1F497D',
            'bg_color': '#C5D9F1',
            'align': 'center'  ,
            'border': 1
        })
        self.dict_cell_format['BG_FONT_BLUE_H3'] = self.BG_FONT_BLUE_H3
        self.dict_cell_format['BLUE_H3'] = self.BG_FONT_BLUE_H3

        self.BG_FONT_BLUE_H2 = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 16,
            'font_color': '#1F497D',
            'bg_color': '#C5D9F1',
            'align': 'center'  ,
            'border': 1
        })
        self.dict_cell_format['BG_FONT_BLUE_H2'] = self.BG_FONT_BLUE_H2
        self.dict_cell_format['BLUE_H2'] = self.BG_FONT_BLUE_H2

        self.BG_FONT_BLUE_H1 = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 18,
            'font_color': '#1F497D',
            'bg_color': '#C5D9F1',
            'align': 'center'  ,
            'border': 1
        })
        self.dict_cell_format['BG_FONT_BLUE_H1'] = self.BG_FONT_BLUE_H1
        self.dict_cell_format['BLUE_H1'] = self.BG_FONT_BLUE_H1

        ########## Colored Title Formatting (END) #################################
        
        self.SECTION_HEADER_PERCENTAGE = self.workbook.add_format({
            'bold': True,
            'font_name': 'Calibri',
            'font_size': 11,
            'font_color': '#1F497D',
            'bg_color': '#C5D9F1',
            'align': 'left',
            'valign': 'vcenter',
            'text_wrap': True,
            'border': 1,
            'num_format': '0.0000%',
        })
        self.dict_cell_format['SECTION_HEADER_PERCENTAGE'] = self.SECTION_HEADER_PERCENTAGE  
        self.dict_cell_format['SECTION%'] = self.SECTION_HEADER_PERCENTAGE
        
        self.TEXT_RIGHT_JUSTIFY = self.workbook.add_format({
            'font_name': 'Calibri',
            'font_size': 11,
            'align': 'right'
        })
        self.dict_cell_format['TEXT_RIGHT_JUSTIFY'] = self.TEXT_RIGHT_JUSTIFY    
        self.dict_cell_format['TEXT_RIGHT'] = self.TEXT_RIGHT_JUSTIFY

        ################# Number Formatting #################
        self.NUM_COMMA = self.workbook.add_format({'num_format': '#,##0', 'border': 1})
        self.dict_cell_format['NUM_COMMA'] = self.NUM_COMMA  
        self.dict_cell_format['NUM,'] = self.NUM_COMMA

        self.NUM_PERCENTAGE = self.workbook.add_format({'num_format': "0.0000%", 'border': 1})
        self.dict_cell_format['NUM_PERCENTAGE'] = self.NUM_PERCENTAGE  
        self.dict_cell_format['NUM%.4'] = self.NUM_PERCENTAGE

        self.NUM_PERCENTAGE = self.workbook.add_format({'num_format': "0.000%", 'border': 1})
        self.dict_cell_format['NUM_PERCENTAGE'] = self.NUM_PERCENTAGE  
        self.dict_cell_format['NUM%.3'] = self.NUM_PERCENTAGE

        self.NUM_PERCENTAGE = self.workbook.add_format({'num_format': "0.00%", 'border': 1})
        self.dict_cell_format['NUM_PERCENTAGE'] = self.NUM_PERCENTAGE  
        self.dict_cell_format['NUM%.2'] = self.NUM_PERCENTAGE

        self.NUM_PERCENTAGE = self.workbook.add_format({'num_format': "0.0%", 'border': 1})
        self.dict_cell_format['NUM_PERCENTAGE'] = self.NUM_PERCENTAGE  
        self.dict_cell_format['NUM%.1'] = self.NUM_PERCENTAGE


        ################# Highlight Formatting #################
        self.BG_LIGHT_YELLOW = self.workbook.add_format({"bg_color": "fcfc81"})
        self.dict_cell_format['BG_LIGHT_YELLOW'] = self.BG_LIGHT_YELLOW  
        self.dict_cell_format['YELLOW_BG'] = self.BG_LIGHT_YELLOW

    def add_new_format(self, format_dict, format_name):
        """ Add a new user-defined format. """
        if format_name not in self.dict_cell_format.keys():
            self.dict_cell_format[format_name] = self.workbook.add_format(format_dict)
            return 0
        logger.info("ERROR: Failed to add new format, the format name has already existed!")
        return 1