import base64
import random
import pandas as pd
import numpy as np

class TextEncryptor:
    """
    基于XOR算法的文本加密解密工具类。

    该类提供文本加密和解密功能，支持单个字符串以及整个Pandas DataFrame的加解密操作。
    加密后的数据使用Base64 URL安全编码，便于存储和传输。

    Attributes:
        key (str): 加密解密使用的密钥。如果为None，则使用空字符串作为密钥。
        suffix (str): DataFrame列名加密后的后缀，默认为'_encrypted'。

    Example:
        >>> encryptor = TextEncryptor(key="my_secret_key")
        >>> encrypted = encryptor.encrypt("Hello World")
        >>> decrypted = encryptor.decrypt(encrypted)
        >>> print(decrypted)  # 输出: Hello World
    """

    def __init__(self, key=None, suffix='_encrypted'):
        """
        初始化加密器实例。

        Parameters:
            key (str, optional): 加密解密使用的密钥。如果为None，则使用空字符串作为密钥。
                               注意：使用空密钥加密后的数据将不具有保密性。
            suffix (str, optional): 当对DataFrame进行加密时，列名添加的后缀。
                                  默认为'_encrypted'。解密时会移除此后缀。
        """
        self.key = key
        self.suffix = suffix

    def encrypt(self, text):
        """
        对输入的文本进行加密。

        使用XOR算法将明文与密钥进行异或操作，然后通过Base64 URL安全编码输出。
        加密结果包含原始文本长度信息（前2个字节），用于解密时的验证。

        Parameters:
            text (str): 需要加密的明文字符串。

        Returns:
            str: 加密后的字符串，使用Base64 URL安全编码。

        Raises:
            AttributeError: 如果key属性为None（self.key为None时，实际使用空字符串）。

        Example:
            >>> encryptor = TextEncryptor(key="secret")
            >>> encrypted = encryptor.encrypt("Hello")
            >>> print(encrypted)  # 输出类似: aAAAAS垂涎==
        """
        # Text to bytes
        text_bytes = text.encode('utf-8')

        # Expand byte length
        key_bytes = (self.key * (len(text_bytes) // len(self.key) + 1)).encode('utf-8')
        key_bytes = key_bytes[:len(text_bytes)]

        # XOR encryption
        encrypted_bytes = bytes([text_bytes[i] ^ key_bytes[i] for i in range(len(text_bytes))])

        # Add 2 more bytes for verification
        length_byte = len(text).to_bytes(2, 'big')

        # combine
        result_bytes = length_byte + encrypted_bytes
        return base64.urlsafe_b64encode(result_bytes).decode('utf-8')

    def decrypt(self, encrypted_text):
        """
        对加密后的文本进行解密。

        首先使用Base64解码，然后提取长度信息（前2字节），接着使用XOR算法与密钥进行异或操作恢复明文。
        解密后会验证恢复文本的长度是否与存储的长度信息匹配，以确保数据完整性。

        Parameters:
            encrypted_text (str): 经过encrypt方法加密的Base64编码字符串。

        Returns:
            str: 解密后的原始明文字符串。

        Raises:
            ValueError: 如果解密失败，可能原因包括：
                       - Base64解码失败（输入不是有效的Base64字符串）
                       - 长度验证失败（数据被篡改或使用了不同的密钥）
                       - 其他解码错误

        Example:
            >>> encryptor = TextEncryptor(key="secret")
            >>> encrypted = encryptor.encrypt("Hello")
            >>> decrypted = encryptor.decrypt(encrypted)
            >>> print(decrypted)  # 输出: Hello
        """
        try:
            # b64 decryption
            decoded_bytes = base64.urlsafe_b64decode(encrypted_text.encode('utf-8'))

            # extract length info
            text_length = int.from_bytes(decoded_bytes[:2], 'big')

            # extract encryption info
            encrypted_bytes = decoded_bytes[2:]

            # regenerate key
            key_bytes = (self.key * (len(encrypted_bytes) // len(self.key) + 1)).encode('utf-8')
            key_bytes = key_bytes[:len(encrypted_bytes)]

            # XOR Decrypt
            decrypted_bytes = bytes([encrypted_bytes[i] ^ key_bytes[i] for i in range(len(encrypted_bytes))])

            # Check Length
            if len(decrypted_bytes) != text_length:
                raise ValueError("Text Length Does Not Match!")

            return decrypted_bytes.decode('utf-8')
        except:
            raise ValueError("Decrypt Failed! Data Might be Destroyed or Incorrect Key!")

    def encrypt_dataframe(self, data):
        """
        对整个Pandas DataFrame进行加密。

        将DataFrame中的所有列值转换为字符串格式后进行加密，同时为列名添加指定的后缀。
        该方法返回一个全新的DataFrame，原始数据不会被修改。

        Parameters:
            data (pandas.DataFrame): 需要加密的Pandas DataFrame对象。
                                   所有列的值都会被转换为字符串格式进行加密。

        Returns:
            pandas.DataFrame: 加密后的新DataFrame，具有以下特点：
                             - 所有列值都经过加密，使用Base64编码
                             - 所有列名都添加了初始化时指定的后缀（默认为'_encrypted'）
                             - 返回的是副本，原始DataFrame保持不变

        Raises:
            AttributeError: 如果key属性为None导致加密失败。

        Note:
            - 加密后的DataFrame无法直接用于数据分析，必须先解密
            - 建议在加密前备份原始DataFrame的列名对应关系

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
            >>> encryptor = TextEncryptor(key="secret")
            >>> encrypted_df = encryptor.encrypt_dataframe(df)
            >>> print(encrypted_df.columns.tolist())  # 输出: ['name_encrypted', 'age_encrypted']
        """
        res = data.copy()
        collist = data.columns.tolist()
        for col in collist:
            ## Encryption
            res[col] = res[col].astype(str)
            res[col] = res[col].apply(lambda x: self.encrypt(x))
        res.columns = [x + self.suffix for x in res.columns]
        return res

    def decrypt_dataframe(self, data):
        """
        对加密后的Pandas DataFrame进行解密。

        遍历DataFrame中的所有列，对每个列值进行解密，同时移除列名中的加密后缀。
        该方法返回一个全新的DataFrame，原始数据不会被修改。

        Parameters:
            data (pandas.DataFrame): 需要解密的Pandas DataFrame对象。
                                   应该是由encrypt_dataframe方法加密产生的DataFrame。

        Returns:
            pandas.DataFrame: 解密后的新DataFrame，具有以下特点：
                             - 所有列值都经过解密，恢复为原始字符串格式
                             - 所有列名都移除了初始化时指定的后缀（默认为'_encrypted'）
                             - 返回的是副本，原始DataFrame保持不变

        Raises:
            ValueError: 如果解密失败，可能原因包括：
                       - 列值不是有效的加密字符串
                       - 使用了错误的密钥进行解密
                       - 数据在传输或存储过程中被损坏
            UnicodeDecodeError: 如果解密后的字节无法正确解码为UTF-8字符串。

        Note:
            - 加密和解密必须使用相同的密钥
            - 如果DataFrame包含非加密的列，解密操作可能会失败

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'name_encrypted': ['aGVsbG8=', 'd29ybGQ='],
            ...                    'age_encrypted': ['c2F2ZWQ=', 'dGVzdA==']})
            >>> encryptor = TextEncryptor(key="secret")
            >>> decrypted_df = encryptor.decrypt_dataframe(df)
            >>> print(decrypted_df.columns.tolist())  # 输出: ['name', 'age']
        """
        res = data.copy()
        collist = data.columns.tolist()
        for col in collist:
            ## Encryption
            res[col] = res[col].apply(lambda x: self.decrypt(x))
        res.columns = [x.replace(self.suffix, "") for x in res.columns]
        return res
