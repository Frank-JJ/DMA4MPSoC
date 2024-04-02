-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_4_weights_V_29_rom is 
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


architecture rtl of infer_layer_4_weights_V_29_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "11111110111111", 1 => "11000010110000", 2 => "11010010111000", 
    3 => "11110100001110", 4 => "10110110010111", 5 => "00100010101100", 
    6 => "00111100101000", 7 => "01001011111000", 8 => "00000111010101", 
    9 => "11001100100011", 10 => "00010100100100", 11 => "10111110110111", 
    12 => "10011101010100", 13 => "00000101000001", 14 => "01100110011111", 
    15 => "01010100000101", 16 => "00010100110000", 17 => "10010110111100", 
    18 => "10010110110010", 19 => "00000100110111", 20 => "10110011100100", 
    21 => "01011010010100", 22 => "00011101100100", 23 => "01010001010010", 
    24 => "11000111110001", 25 => "01001100010111", 26 => "11010100000010", 
    27 => "00000011101010", 28 => "10100111111100", 29 => "10011001000101", 
    30 => "10010111110000", 31 => "11110011010101", 32 => "11111110111000", 
    33 => "11100011100010", 34 => "00001100011011", 35 => "01011001111000", 
    36 => "11010100100101", 37 => "00111101111111", 38 => "10100000101111", 
    39 => "00101011010011", 40 => "11100001111111", 41 => "01011010101000", 
    42 => "11001011010001", 43 => "10011100011110", 44 => "01010001010110", 
    45 => "01100001010101", 46 => "10101100110100", 47 => "00101000011101", 
    48 => "00110001000010", 49 => "00101101100111", 50 => "10101011111001", 
    51 => "01001010010101", 52 => "10011011101111", 53 => "10100111100101", 
    54 => "00001101111101", 55 => "00001101101010", 56 => "11111100110111", 
    57 => "00111101001010", 58 => "10101010011000", 59 => "01001011111010", 
    60 => "10111000100011", 61 => "00110000010101", 62 => "00011100101110", 
    63 => "10100011010001", 64 => "01011001010111", 65 => "11000001010010", 
    66 => "11000100001110", 67 => "00111000100010", 68 => "11001110010110", 
    69 => "00010110111001", 70 => "01010010001111", 71 => "11010100100110", 
    72 => "10100011111111", 73 => "00111101100110", 74 => "01010111000000", 
    75 => "00111000110011", 76 => "11010111001011", 77 => "11100111000101", 
    78 => "00111011100011", 79 => "00101001010111", 80 => "00011111100101", 
    81 => "11100010101101", 82 => "01001111111001", 83 => "11010010000010", 
    84 => "11001000000010", 85 => "11110101010110", 86 => "00111111000101", 
    87 => "00111111000010", 88 => "00111010101010", 89 => "00011101111011", 
    90 => "10011101110010", 91 => "10111111110110", 92 => "11101110111111", 
    93 => "10011111000110", 94 => "10110011100101", 95 => "01010011111101", 
    96 => "11010011101110", 97 => "00001000110100", 98 => "00011011111101", 
    99 => "00000001000111", 100 => "11111000101011", 101 => "10101001010000", 
    102 => "00000100101011", 103 => "11011111000000", 104 => "00100001111010", 
    105 => "00010010111001", 106 => "10100111010010", 107 => "11110010100000", 
    108 => "11001111010101", 109 => "01010101011011", 110 => "11001111010100", 
    111 => "11010000101110", 112 => "11001101110000", 113 => "01001011001011", 
    114 => "00010100100000", 115 => "11000110111011", 116 => "01011111011001", 
    117 => "01010001000101", 118 => "00011101111111", 119 => "10101011101101", 
    120 => "00101101000000", 121 => "01000101011111", 122 => "01010100110111", 
    123 => "01010001011001", 124 => "00000101111100", 125 => "00110100101111", 
    126 => "10111000101000", 127 => "01100100100111", 128 => "00000010011001", 
    129 => "11101000010100", 130 => "01011111011110", 131 => "10110100010110", 
    132 => "11001111001001", 133 => "11110010110111", 134 => "11010001001000", 
    135 => "11001111001011", 136 => "00010110110010", 137 => "10100011010011", 
    138 => "11010111100111", 139 => "01010100111110", 140 => "11011011011100", 
    141 => "10011110110110", 142 => "11000111011001", 143 => "01001111110110", 
    144 => "00110000101010", 145 => "10100010110110", 146 => "01100010100011", 
    147 => "01010000110001", 148 => "10111001110010", 149 => "10111001011000", 
    150 => "00110011011001", 151 => "00001001100110", 152 => "01000110011010", 
    153 => "00011000011000", 154 => "11001101111101", 155 => "11001110111010", 
    156 => "10101000111110", 157 => "11111100011100", 158 => "00000001000011", 
    159 => "11110100010110", 160 => "10101010111011", 161 => "00101000010011", 
    162 => "10101101000100", 163 => "00100101101101", 164 => "01010010110110", 
    165 => "01100010000101", 166 => "11000000000100", 167 => "11111110011101", 
    168 => "00111001000011", 169 => "01100110011001", 170 => "10110101001100", 
    171 => "11001011110000", 172 => "10100100011101", 173 => "00011101001111", 
    174 => "00001101011000", 175 => "11111101101010", 176 => "00010001101101", 
    177 => "10110111001010", 178 => "11011100010010", 179 => "01100100000000", 
    180 => "01001010100011", 181 => "00111000100010", 182 => "11110011101001", 
    183 => "11011100001100", 184 => "01011100101000", 185 => "10111100011001", 
    186 => "01011100001011", 187 => "01100000110110", 188 => "11010101011011", 
    189 => "11010111000011", 190 => "11100010000010", 191 => "00111100010110", 
    192 => "11001000010111", 193 => "11011001010101", 194 => "10011011001111", 
    195 => "00110000101111", 196 => "01000010101000", 197 => "10100001110110", 
    198 => "01010110001011", 199 => "10111011111000", 200 => "01001000111001", 
    201 => "00101001101010", 202 => "11000101010100", 203 => "00111000001111", 
    204 => "10111001010110", 205 => "11010010100010", 206 => "00100110000110", 
    207 => "11001011110100", 208 => "01100110011101", 209 => "11100001001110", 
    210 => "01000100100011", 211 => "11001101101110", 212 => "10110101010110", 
    213 => "11101010101111", 214 => "11010011001110", 215 => "10011000111001", 
    216 => "00010010001000", 217 => "10101111100111", 218 => "00001011000110", 
    219 => "10101111101001", 220 => "01100101000001", 221 => "00001000001101", 
    222 => "00100000101111", 223 => "10101111111011", 224 => "01010001111110", 
    225 => "10100000000111", 226 => "00001001101100", 227 => "01000000111110", 
    228 => "00111010110101", 229 => "00000011011111", 230 => "00111010011010", 
    231 => "11100001000011", 232 => "11000011000110", 233 => "10110110101000", 
    234 => "11111111101001", 235 => "10011000101000", 236 => "01100010100001", 
    237 => "11001011010100", 238 => "00101101001101", 239 => "11101111011111", 
    240 => "01001011110011", 241 => "11000010100000", 242 => "10101001110011", 
    243 => "00011110010000", 244 => "00100111011000", 245 => "01010111000010", 
    246 => "11010100000101", 247 => "01010000010011", 248 => "10111101100110", 
    249 => "01001010000001", 250 => "00010111111000", 251 => "00011111001111", 
    252 => "10101000011000", 253 => "11110100011100", 254 => "11100100101001", 
    255 => "00011110111011", 256 => "11010010100001", 257 => "01000100001001", 
    258 => "11111000101000", 259 => "00110111101011", 260 => "10101101000101", 
    261 => "11011011001110", 262 => "00110100000111", 263 => "10110101000100", 
    264 => "11101010100100", 265 => "01100110110000", 266 => "10101001011011", 
    267 => "00111001100100", 268 => "10110011101110", 269 => "00000111100100", 
    270 => "01011110101101", 271 => "00101011000011", 272 => "01100101010001", 
    273 => "11001001100100", 274 => "11100000000110", 275 => "00100010000000", 
    276 => "10011110010100", 277 => "01001010110011", 278 => "01000011100010", 
    279 => "00111000110111", 280 => "11001011110001", 281 => "11101001101110", 
    282 => "00100101100100", 283 => "00101011000001", 284 => "10111100100100", 
    285 => "11011010010100", 286 => "11101111111000", 287 => "00011100010011" );


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

entity infer_layer_4_weights_V_29 is
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

architecture arch of infer_layer_4_weights_V_29 is
    component infer_layer_4_weights_V_29_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_4_weights_V_29_rom_U :  component infer_layer_4_weights_V_29_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


