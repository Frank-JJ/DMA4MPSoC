-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_6_weights_V_6_rom is 
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


architecture rtl of infer_layer_6_weights_V_6_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "00001110011000", 1 => "01100001001000", 2 => "11111110110110", 
    3 => "11110001111000", 4 => "00000001100010", 5 => "01010111111010", 
    6 => "00011100000100", 7 => "10101011111110", 8 => "11100101110011", 
    9 => "00001110011011", 10 => "11101011001110", 11 => "00011000011111", 
    12 => "11010101101011", 13 => "00110111010001", 14 => "11011100110000", 
    15 => "01010001110111", 16 => "10100110011010", 17 => "01010010010100", 
    18 => "11011101111011", 19 => "10110011000001", 20 => "11001110111011", 
    21 => "01100100101100", 22 => "11001001000101", 23 => "11000010101011", 
    24 => "00100010000000", 25 => "00100100101110", 26 => "00101000010100", 
    27 => "10100001000011", 28 => "01001110011110", 29 => "11110101100100", 
    30 => "00110011011111", 31 => "00010000101111", 32 => "00111000011010", 
    33 => "10111011100110", 34 => "00111111101100", 35 => "00001101001010", 
    36 => "00000010000111", 37 => "01010010100101", 38 => "11010101101000", 
    39 => "01000100010100", 40 => "10101011110100", 41 => "11001100010111", 
    42 => "11010101101111", 43 => "00100101110010", 44 => "10100000011000", 
    45 => "01011000111010", 46 => "00010100000111", 47 => "00101101100111", 
    48 => "00011111000000", 49 => "01001101110011", 50 => "00100101001001", 
    51 => "11001110000110", 52 => "10100001111001", 53 => "00010111111000", 
    54 => "00000010011110", 55 => "00000111011000", 56 => "10111001001000", 
    57 => "00000111010011", 58 => "00010111110100", 59 => "10011111011001", 
    60 => "00000100110110", 61 => "00111000010001", 62 => "01000111101010", 
    63 => "00111100000101", 64 => "00101000111011", 65 => "11010110011001", 
    66 => "00111110111010", 67 => "00011100010110", 68 => "00100110110100", 
    69 => "11110010000001", 70 => "00110010100100", 71 => "10011010001111", 
    72 => "11001010010000", 73 => "11100010101101", 74 => "00000010010000", 
    75 => "10111101010010", 76 => "00001011111001", 77 => "10100000001111", 
    78 => "00111110000001", 79 => "00001111110101", 80 => "11000010111110", 
    81 => "11101011001001", 82 => "01011011000010", 83 => "01011111001000", 
    84 => "10111011001101", 85 => "01001111111000", 86 => "11101001000110", 
    87 => "10110100110001", 88 => "01001010001100", 89 => "11101111110101", 
    90 => "00001011010100", 91 => "00101101001001", 92 => "01010001000110", 
    93 => "01000010001100", 94 => "11001011100011", 95 => "10111001110110", 
    96 => "11111111010010", 97 => "00011101111011", 98 => "01000001101011", 
    99 => "11110101101010", 100 => "10100110101001", 101 => "00010100111001", 
    102 => "10100000010010", 103 => "11011001010010", 104 => "11011111110010", 
    105 => "11011101111000", 106 => "11010101101010", 107 => "00101000011000", 
    108 => "11101111011011", 109 => "11011101111101", 110 => "00101111101001", 
    111 => "00111110000010", 112 => "11001010010111", 113 => "11010001001101", 
    114 => "11101100010110", 115 => "00011011010010", 116 => "00000111011001", 
    117 => "10100000000001", 118 => "00001111010000", 119 => "11100100111011", 
    120 => "01011011101011", 121 => "01011000000010", 122 => "00001101111110", 
    123 => "11100110000000", 124 => "01011011101111", 125 => "01000111101010", 
    126 => "10110101010011", 127 => "11111010101001", 128 => "11100111111001", 
    129 => "11010111011000", 130 => "11000000010111", 131 => "00011010011010", 
    132 => "11100111011001", 133 => "11000001000010", 134 => "10110011001011", 
    135 => "00101001011001", 136 => "00111100010011", 137 => "01010111100000", 
    138 => "01000101011010", 139 => "11000001010110", 140 => "00011011000111", 
    141 => "00011101011110", 142 => "11010001001111", 143 => "00000101101111", 
    144 => "10101110110100", 145 => "10110101010010", 146 => "00111100100110", 
    147 => "01100111100100", 148 => "11001100011111", 149 => "11111111010101", 
    150 => "11001000010000", 151 => "01000010110111", 152 => "11100111001111", 
    153 => "00111001110100", 154 => "10100010010111", 155 => "00011011010000", 
    156 => "00011011011110", 157 => "00110001000001", 158 => "11000000100011", 
    159 => "01011011000111", 160 => "11111101111100", 161 => "10110010101001", 
    162 => "11111100111101", 163 => "10110100101011", 164 => "11011110101011", 
    165 => "10100000101110", 166 => "11001110111011", 167 => "00001111100100", 
    168 => "01100001010110", 169 => "11101100000111", 170 => "11100011111001", 
    171 => "11110001011100", 172 => "11101111110110", 173 => "10011011111111", 
    174 => "10110011011010", 175 => "00001100011101", 176 => "10101000010111", 
    177 => "11111000010110", 178 => "00011011001000", 179 => "11111100001110", 
    180 => "01100011001100", 181 => "11010011111000", 182 => "01001011100001", 
    183 => "00100011110111", 184 => "11010100000110", 185 => "01000110010000", 
    186 => "11111001111000", 187 => "10110000101011", 188 => "00101110011000", 
    189 => "11001011110111", 190 => "01001101100010", 191 => "01100011110100", 
    192 => "00100100001011", 193 => "00100011101011", 194 => "10011110111100", 
    195 => "01001000101000", 196 => "00010011000111", 197 => "00001110111000", 
    198 => "01010110101001", 199 => "01011011011111", 200 => "00110100010011", 
    201 => "11011110010010", 202 => "01010101101111", 203 => "11001000100110", 
    204 => "11110011000000", 205 => "11100011010000", 206 => "11110001110110", 
    207 => "00001111000101", 208 => "11111001110111", 209 => "11100111111111", 
    210 => "00100010011111", 211 => "11110110110001", 212 => "00010011100010", 
    213 => "00110101110100", 214 => "10100000010101", 215 => "00101111011001", 
    216 => "01000101110100", 217 => "10111111000111", 218 => "00111000011101", 
    219 => "10101001101101", 220 => "10111100010101", 221 => "01011111100011", 
    222 => "11101100111101", 223 => "11001010000010", 224 => "10101101011010", 
    225 => "01010011000010", 226 => "10100001101101", 227 => "10011111001010", 
    228 => "11110110011100", 229 => "01010100111000", 230 => "10101010011011", 
    231 => "10101111001111", 232 => "01001110001000", 233 => "10101100000100", 
    234 => "10011000100000", 235 => "01010000100110", 236 => "00010010001100", 
    237 => "10010101001011", 238 => "00100001101010", 239 => "01001111010110", 
    240 => "11100001000000", 241 => "10100111111111", 242 => "11001010100101", 
    243 => "00110000000010", 244 => "00111111101010", 245 => "10111011010010", 
    246 => "01000100100101", 247 => "00101111000111", 248 => "11101000101111", 
    249 => "00110001111111", 250 => "11010111101001", 251 => "11111100111001", 
    252 => "11100100011000", 253 => "01010001011100", 254 => "11001111000011", 
    255 => "00110110111000", 256 => "11001100100000", 257 => "01000101001000", 
    258 => "11100011010001", 259 => "00000100110010", 260 => "10101110001111", 
    261 => "11111001100001", 262 => "11000011101001", 263 => "00110101000101", 
    264 => "00110011111101", 265 => "10111100001010", 266 => "10110100111000", 
    267 => "01010111011101", 268 => "00000011111010", 269 => "10111100000101", 
    270 => "00110101111101", 271 => "01011010011110", 272 => "10100011100000", 
    273 => "01001010010101", 274 => "10101000100110", 275 => "11110110111001", 
    276 => "11011101111111", 277 => "00001100100100", 278 => "00010101010001", 
    279 => "10011111001011", 280 => "11010000000110", 281 => "11011101101010", 
    282 => "11000111001111", 283 => "00010010111010", 284 => "10100010101001", 
    285 => "11111000100010", 286 => "10110010110010", 287 => "10111110101000" );


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

entity infer_layer_6_weights_V_6 is
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

architecture arch of infer_layer_6_weights_V_6 is
    component infer_layer_6_weights_V_6_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_6_weights_V_6_rom_U :  component infer_layer_6_weights_V_6_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


