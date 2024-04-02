-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_4_weights_V_3_rom is 
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


architecture rtl of infer_layer_4_weights_V_3_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "10100000111100", 1 => "00011010011101", 2 => "00010111000110", 
    3 => "01000111010011", 4 => "11000001110001", 5 => "00111111101101", 
    6 => "01101000010111", 7 => "10111100011011", 8 => "10100101101111", 
    9 => "11111110110110", 10 => "11101111001011", 11 => "10111100100111", 
    12 => "01000101010011", 13 => "11011101111011", 14 => "11010111001101", 
    15 => "10101010010100", 16 => "00010011010100", 17 => "01011111001000", 
    18 => "01001011000101", 19 => "01010001100100", 20 => "01010000111000", 
    21 => "11001001101010", 22 => "00100111001000", 23 => "01011010110100", 
    24 => "11000100111110", 25 => "00001111100101", 26 => "11110111111111", 
    27 => "00111101010010", 28 => "00111100110100", 29 => "00100010011010", 
    30 => "10100010101001", 31 => "01100000100011", 32 => "00000100000110", 
    33 => "00111001001001", 34 => "11011011000000", 35 => "01000000101101", 
    36 => "01000011101001", 37 => "11001010000001", 38 => "11001011010111", 
    39 => "11100011001000", 40 => "00011110011100", 41 => "01000100100101", 
    42 => "00100100001011", 43 => "00001001101000", 44 => "01011100001110", 
    45 => "11111100000001", 46 => "11010101000100", 47 => "10111000101010", 
    48 => "11001001010010", 49 => "00111000011001", 50 => "00111000111011", 
    51 => "10010011011100", 52 => "00010011010101", 53 => "10110010011101", 
    54 => "10100101010110", 55 => "00011000100101", 56 => "10110111100101", 
    57 => "01011111001011", 58 => "11001101101101", 59 => "10101110000000", 
    60 => "11101001000010", 61 => "11011001110000", 62 => "11110001111001", 
    63 => "11010000111111", 64 => "11001000100001", 65 => "11101110110000", 
    66 => "00010000010111", 67 => "10111000001101", 68 => "00100101011000", 
    69 => "11111001100001", 70 => "10100010111111", 71 => "00001100100101", 
    72 => "01000011111110", 73 => "00101111100000", 74 => "00010011001011", 
    75 => "01011000010010", 76 => "00011110110001", 77 => "11000010001111", 
    78 => "00010110011111", 79 => "00100101110101", 80 => "00111110000111", 
    81 => "11100101010110", 82 => "10100000000011", 83 => "10101100100000", 
    84 => "11001111101100", 85 => "00101111010110", 86 => "01011101001011", 
    87 => "00010101000011", 88 => "10111011101010", 89 => "10011010001011", 
    90 => "10101011111001", 91 => "00110010100011", 92 => "11001011000110", 
    93 => "11010111110110", 94 => "00100111011001", 95 => "00011010011101", 
    96 => "10111010010111", 97 => "11001010001100", 98 => "01010111101011", 
    99 => "11111110000011", 100 => "00111001011111", 101 => "10011011001010", 
    102 => "01011010000010", 103 => "00101100110010", 104 => "10010010110111", 
    105 => "01011110100110", 106 => "00101100010110", 107 => "00000011011101", 
    108 => "01011111111010", 109 => "11101000001011", 110 => "11001100010101", 
    111 => "11001100110000", 112 => "10101110101011", 113 => "10100000100101", 
    114 => "10101110100100", 115 => "11000011011010", 116 => "11011010011111", 
    117 => "01010011011100", 118 => "00101111010100", 119 => "10110011011110", 
    120 => "11110010011101", 121 => "00001101101101", 122 => "00111000100001", 
    123 => "11100000110110", 124 => "00011101101010", 125 => "01010101010110", 
    126 => "00010011101110", 127 => "00111101111110", 128 => "11011101111101", 
    129 => "11100100110110", 130 => "01000101010011", 131 => "00110000111000", 
    132 => "00111100100000", 133 => "00000011011100", 134 => "00111000010111", 
    135 => "11001110000010", 136 => "00101111111100", 137 => "11111010001010", 
    138 => "00010010110011", 139 => "01100001111011", 140 => "11110110111001", 
    141 => "01000011001011", 142 => "00000110101100", 143 => "10010110101000", 
    144 => "10100110011001", 145 => "11100011110100", 146 => "01010010001110", 
    147 => "11000001010001", 148 => "11101111000100", 149 => "00110111010011", 
    150 => "11010001001110", 151 => "11011010101100", 152 => "10011111111110", 
    153 => "11011001001100", 154 => "10111011110000", 155 => "00110110110011", 
    156 => "11011001100101", 157 => "10011001111101", 158 => "00010000101001", 
    159 => "10111100100101", 160 => "11101001000111", 161 => "11111001010110", 
    162 => "00000111110001", 163 => "01000010000101", 164 => "00111000001110", 
    165 => "00010110110111", 166 => "10110100111010", 167 => "10101010101100", 
    168 => "11000110110001", 169 => "11100101111101", 170 => "00101001100011", 
    171 => "00100000110101", 172 => "11111101100111", 173 => "11100111010110", 
    174 => "00101010111100", 175 => "01100001100111", 176 => "00100001001101", 
    177 => "01000011101101", 178 => "00100011110111", 179 => "11010110000010", 
    180 => "10111000001010", 181 => "10111110010101", 182 => "10110011001111", 
    183 => "10011010010011", 184 => "10110111110111", 185 => "00010110001110", 
    186 => "11111000100000", 187 => "01000111111111", 188 => "11011011001000", 
    189 => "11000001101111", 190 => "00101101111001", 191 => "11000100111001", 
    192 => "10101010101010", 193 => "11011100100000", 194 => "11011111100000", 
    195 => "10010110101010", 196 => "10100100011010", 197 => "10100101000000", 
    198 => "00111001010010", 199 => "01100100010010", 200 => "11001110110100", 
    201 => "00001010001011", 202 => "00101001000100", 203 => "11001011010000", 
    204 => "01001100111111", 205 => "10111111100100", 206 => "00001010001110", 
    207 => "00001100001011", 208 => "01100101001110", 209 => "00001011000010", 
    210 => "00011001010011", 211 => "01010101100011", 212 => "11000101001101", 
    213 => "10101100001111", 214 => "01001011110111", 215 => "01001111100111", 
    216 => "00000100101000", 217 => "00110000100000", 218 => "11001110111000", 
    219 => "00101010101011", 220 => "00010100000111", 221 => "11101001000010", 
    222 => "10011011000010", 223 => "01011010011100", 224 => "10100000101100", 
    225 => "10100010011101", 226 => "11101100011000", 227 => "11010111001000", 
    228 => "11010001001001", 229 => "00000010011010", 230 => "00010100011001", 
    231 => "11110110100111", 232 => "10110011111100", 233 => "11001011010000", 
    234 => "01011001001010", 235 => "01000100000001", 236 => "01011110101011", 
    237 => "01011010101010", 238 => "00111010000101", 239 => "10010110101010", 
    240 => "11101110111000", 241 => "11101110011101", 242 => "10010110110100", 
    243 => "00110111110011", 244 => "00100101011111", 245 => "11011000000101", 
    246 => "11001001111010", 247 => "00000000101111", 248 => "10010111101010", 
    249 => "00101100111011", 250 => "11101010110010", 251 => "11111101100110", 
    252 => "11110100011000", 253 => "10111101101011", 254 => "10011100000110", 
    255 => "11101001000110", 256 => "00011000011010", 257 => "10011000111110", 
    258 => "10100011011011", 259 => "11011011011000", 260 => "01011111000010", 
    261 => "00000011101101", 262 => "00100111110001", 263 => "00111010000010", 
    264 => "10011111101110", 265 => "11100010011001", 266 => "00010100101110", 
    267 => "00001011101110", 268 => "00010000011101", 269 => "00100011000111", 
    270 => "01010001000111", 271 => "11000001100101", 272 => "00111011010110", 
    273 => "11110111101100", 274 => "00011010111001", 275 => "00101000010110", 
    276 => "00110100000110", 277 => "00010000010100", 278 => "10010111111000", 
    279 => "01001010011110", 280 => "01010001111011", 281 => "00101100011101", 
    282 => "11001010001000", 283 => "00000011000000", 284 => "01010100001111", 
    285 => "10101000100110", 286 => "10111010010001", 287 => "00101100100000" );


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

entity infer_layer_4_weights_V_3 is
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

architecture arch of infer_layer_4_weights_V_3 is
    component infer_layer_4_weights_V_3_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_4_weights_V_3_rom_U :  component infer_layer_4_weights_V_3_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


