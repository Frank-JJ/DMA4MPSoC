-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_6_weights_V_11_rom is 
    generic(
             DWIDTH     : integer := 14; 
             AWIDTH     : integer := 9; 
             MEM_SIZE    : integer := 288
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of infer_layer_6_weights_V_11_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "10011010001110", 1 => "01001010111001", 2 => "10010111100110", 
    3 => "10101010110000", 4 => "00001001100100", 5 => "11011010001110", 
    6 => "00010000101000", 7 => "00111101001010", 8 => "10011011110111", 
    9 => "01010010000110", 10 => "11111010010011", 11 => "11000001101011", 
    12 => "10110100111110", 13 => "00011100010110", 14 => "00100100100111", 
    15 => "10011110001000", 16 => "10110011011001", 17 => "01010010111100", 
    18 => "01100100001000", 19 => "01011011111000", 20 => "01010101000110", 
    21 => "00000110001111", 22 => "00001111100101", 23 => "11110111001001", 
    24 => "00000110111110", 25 => "10011111001001", 26 => "10101011101011", 
    27 => "01001110000111", 28 => "00101110101000", 29 => "01001000001011", 
    30 => "11111110101101", 31 => "00001111010110", 32 => "11001100010001", 
    33 => "01011011010111", 34 => "10111011111100", 35 => "01000111111101", 
    36 => "00000011100100", 37 => "01000010100111", 38 => "11000100110011", 
    39 => "00001001101101", 40 => "00100011000010", 41 => "11011000111111", 
    42 => "01100111110100", 43 => "00100010101111", 44 => "00001101011011", 
    45 => "00101011111110", 46 => "11100001011000", 47 => "00101001010010", 
    48 => "10111110111111", 49 => "01001111000100", 50 => "11110000010000", 
    51 => "10111101010011", 52 => "00100111100011", 53 => "10111001001101", 
    54 => "00000101011000", 55 => "00010011000001", 56 => "11011001010101", 
    57 => "00101000010111", 58 => "10110101100100", 59 => "00011110100001", 
    60 => "00111100111001", 61 => "11010011000100", 62 => "11101101010011", 
    63 => "11000100101110", 64 => "00111010111000", 65 => "00001000010110", 
    66 => "11011110011111", 67 => "10101000111100", 68 => "11101110000000", 
    69 => "01001001010110", 70 => "00101011010010", 71 => "00001000110100", 
    72 => "11110100111110", 73 => "11011111100000", 74 => "00101110000101", 
    75 => "11101101010100", 76 => "00110111110100", 77 => "00101101000010", 
    78 => "10101111111101", 79 => "10011000010001", 80 => "11000111110101", 
    81 => "10011111000000", 82 => "10111101100101", 83 => "11101011011111", 
    84 => "00011001110111", 85 => "10100111010111", 86 => "11110101111110", 
    87 => "01010000100101", 88 => "10101110000000", 89 => "11100110001111", 
    90 => "00101101010011", 91 => "00001001010101", 92 => "00000100110011", 
    93 => "01011101000011", 94 => "10101000011110", 95 => "01001101001011", 
    96 => "11010000010001", 97 => "10110010011010", 98 => "00001000100001", 
    99 => "01011101111100", 100 => "11110111001011", 101 => "00010110010011", 
    102 => "11100110000111", 103 => "11100000011101", 104 => "01010011110010", 
    105 => "00011111011010", 106 => "01000110011111", 107 => "00110111011001", 
    108 => "00100101001000", 109 => "10010011101001", 110 => "11100111101100", 
    111 => "01000101000100", 112 => "11111010100101", 113 => "11011000010111", 
    114 => "01001011010000", 115 => "11100000011111", 116 => "00001110101001", 
    117 => "11110101001100", 118 => "10100010111000", 119 => "10110111110110", 
    120 => "01000011000001", 121 => "01101000011100", 122 => "10011110001000", 
    123 => "01001111111011", 124 => "00111110010001", 125 => "11001100111001", 
    126 => "11100110000111", 127 => "11010001011000", 128 => "00111110001011", 
    129 => "10110101110111", 130 => "01011001110100", 131 => "00010111100000", 
    132 => "01100100001111", 133 => "10011101110010", 134 => "00000001001010", 
    135 => "10100011110111", 136 => "01000111110011", 137 => "11111101010011", 
    138 => "01011001000011", 139 => "00011010000110", 140 => "01000001100110", 
    141 => "01001010111101", 142 => "00011110111010", 143 => "00110101110000", 
    144 => "01000100000111", 145 => "10110111001111", 146 => "10100001011011", 
    147 => "10111011001011", 148 => "11111001110001", 149 => "11111011010010", 
    150 => "01001011000000", 151 => "00011110111101", 152 => "11011000100001", 
    153 => "01001100001000", 154 => "00111010100000", 155 => "01011011000101", 
    156 => "00100011001000", 157 => "11011001101001", 158 => "10110011000011", 
    159 => "10111011111110", 160 => "01000001010010", 161 => "01000000101110", 
    162 => "11010011011001", 163 => "00011000101100", 164 => "00111111101011", 
    165 => "11100111111011", 166 => "00011001111001", 167 => "00000111111000", 
    168 => "00010100100010", 169 => "11101000110001", 170 => "11101000010110", 
    171 => "10100101001101", 172 => "11111101001111", 173 => "11000101010011", 
    174 => "00101101110001", 175 => "10101101010000", 176 => "00011001001110", 
    177 => "00011100100110", 178 => "11010100001101", 179 => "10100100010100", 
    180 => "01001010001100", 181 => "00011111010011", 182 => "01010100100111", 
    183 => "00001000110110", 184 => "11101001011010", 185 => "00000000101100", 
    186 => "01001100110100", 187 => "11110101111000", 188 => "01001001111110", 
    189 => "11010000101010", 190 => "00011000010001", 191 => "11011000011110", 
    192 => "00001011111101", 193 => "11111011111100", 194 => "10010110100101", 
    195 => "01001001111001", 196 => "11110000011011", 197 => "11111010000110", 
    198 => "11101011011010", 199 => "11100111111000", 200 => "11010001101011", 
    201 => "01000011110111", 202 => "01011101011110", 203 => "11111110011100", 
    204 => "10111100100010", 205 => "00101110011001", 206 => "00101010000110", 
    207 => "10010100010110", 208 => "11010100101101", 209 => "10111111111001", 
    210 => "11011110010011", 211 => "11100101100111", 212 => "00010111001101", 
    213 => "11100001100000", 214 => "10101101001111", 215 => "11011100010011", 
    216 => "00000111101010", 217 => "00001101110010", 218 => "11011001010111", 
    219 => "10110111000000", 220 => "00011001100111", 221 => "11111100100000", 
    222 => "10011011011000", 223 => "01001000110110", 224 => "01000100110011", 
    225 => "10100000111110", 226 => "11111111000001", 227 => "10111011110010", 
    228 => "10111110110100", 229 => "00011101001011", 230 => "11100000110000", 
    231 => "10101010011000", 232 => "11000110100010", 233 => "10010101110010", 
    234 => "00110110001100", 235 => "00010001100011", 236 => "10110111010110", 
    237 => "11110100100110", 238 => "00101111110010", 239 => "01001100110000", 
    240 => "11010111011010", 241 => "11110011111000", 242 => "00101000001110", 
    243 => "00100110011010", 244 => "10101110111110", 245 => "11110010011100", 
    246 => "11101100110101", 247 => "11110011111011", 248 => "10011011101101", 
    249 => "00010010100011", 250 => "11110101111111", 251 => "10110110110101", 
    252 => "11011010110100", 253 => "00111010010011", 254 => "00110100111101", 
    255 => "01001110101101", 256 => "11010111100011", 257 => "01011001000000", 
    258 => "11001001100111", 259 => "10100100010111", 260 => "10111011110000", 
    261 => "11101001110010", 262 => "10010100101011", 263 => "10100110011110", 
    264 => "11000100011000", 265 => "10101000110111", 266 => "11100010010101", 
    267 => "11100110101111", 268 => "00101110000000", 269 => "00001110111000", 
    270 => "11010111110000", 271 => "11001101011110", 272 => "00101101101000", 
    273 => "11001011111010", 274 => "11000100110100", 275 => "11010001100100", 
    276 => "11111100110010", 277 => "11100101000101", 278 => "00100010111010", 
    279 => "11000101110111", 280 => "00010101101010", 281 => "00101100001010", 
    282 => "00000010011001", 283 => "00111100000010", 284 => "00011000010011", 
    285 => "00000110010101", 286 => "11101110111111", 287 => "11010000000000" );


begin 


memory_access_guard_0: process (addr0) 
begin
      addr0_tmp <= addr0;
--synthesis translate_off
      if (CONV_INTEGER(addr0) > mem_size-1) then
           addr0_tmp <= (others => '0');
      else 
           addr0_tmp <= addr0;
      end if;
--synthesis translate_on
end process;

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= mem(CONV_INTEGER(addr0_tmp)); 
        end if;
    end if;
end process;

end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity infer_layer_6_weights_V_11 is
    generic (
        DataWidth : INTEGER := 14;
        AddressRange : INTEGER := 288;
        AddressWidth : INTEGER := 9);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of infer_layer_6_weights_V_11 is
    component infer_layer_6_weights_V_11_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_6_weights_V_11_rom_U :  component infer_layer_6_weights_V_11_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;

