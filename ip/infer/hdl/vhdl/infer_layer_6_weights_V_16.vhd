-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_6_weights_V_16_rom is 
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


architecture rtl of infer_layer_6_weights_V_16_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "10110101101110", 1 => "11110101100011", 2 => "00101010010110", 
    3 => "11100000000011", 4 => "00100100111011", 5 => "01100110111101", 
    6 => "00001111001010", 7 => "11000000101111", 8 => "01001110010011", 
    9 => "01000101011110", 10 => "11001001010100", 11 => "10100011001010", 
    12 => "11000101000010", 13 => "00010111110001", 14 => "10100110001000", 
    15 => "10101001111010", 16 => "00011001010011", 17 => "11110101100111", 
    18 => "11110010110000", 19 => "11010101011001", 20 => "11011001101110", 
    21 => "01000111100100", 22 => "10111000110100", 23 => "11010011001011", 
    24 => "00110111101000", 25 => "10011011000111", 26 => "00010110011001", 
    27 => "11101010011011", 28 => "11110010011000", 29 => "11000011111001", 
    30 => "11010011100000", 31 => "11010101111000", 32 => "01011000001100", 
    33 => "11011100100100", 34 => "00111100011100", 35 => "00110111011010", 
    36 => "01001101011010", 37 => "10111110010110", 38 => "00110011011100", 
    39 => "10101011100111", 40 => "01010000101010", 41 => "01100000011001", 
    42 => "00100011101101", 43 => "11011101100110", 44 => "11001100101001", 
    45 => "10010011000110", 46 => "10011011000001", 47 => "10010010001001", 
    48 => "10010011011011", 49 => "00000011010000", 50 => "10110001000001", 
    51 => "11011011000101", 52 => "00001100111001", 53 => "10111011011011", 
    54 => "00111010001011", 55 => "01001010011111", 56 => "00010101110100", 
    57 => "00010110000101", 58 => "11100011010010", 59 => "11110011001111", 
    60 => "00010111100100", 61 => "10010111101100", 62 => "01000011101111", 
    63 => "11001101110111", 64 => "00100101110010", 65 => "11101001000010", 
    66 => "00001010100000", 67 => "10100000100100", 68 => "01001101111111", 
    69 => "01001001101101", 70 => "01010101000000", 71 => "11001010001101", 
    72 => "11100000110111", 73 => "01001001101010", 74 => "00111110000010", 
    75 => "11000110100101", 76 => "10011000111010", 77 => "00001001001100", 
    78 => "11101001000100", 79 => "00000111010101", 80 => "10010011011000", 
    81 => "01011011010100", 82 => "10100000101001", 83 => "01001111100011", 
    84 => "11001011100011", 85 => "11000101010000", 86 => "11011011101100", 
    87 => "11100000101101", 88 => "11110101111011", 89 => "11001101000010", 
    90 => "11100001001010", 91 => "00000001111000", 92 => "01011010010100", 
    93 => "00000100000011", 94 => "01011000000000", 95 => "11101110110010", 
    96 => "10101110000101", 97 => "11110011101010", 98 => "11111100000011", 
    99 => "11100010100001", 100 => "00101001100001", 101 => "00110111000010", 
    102 => "01000011111000", 103 => "01000111111010", 104 => "11001100101110", 
    105 => "10101011000001", 106 => "00000101011110", 107 => "01011100000110", 
    108 => "10111011000100", 109 => "10001111101111", 110 => "10111011001000", 
    111 => "11110101011000", 112 => "00010100000101", 113 => "11000100000101", 
    114 => "11000001010010", 115 => "00000110010010", 116 => "10110111011000", 
    117 => "11001100111101", 118 => "11100001110000", 119 => "01001000011110", 
    120 => "11100011000000", 121 => "01011010001101", 122 => "01010000111001", 
    123 => "00011111110000", 124 => "10111111110100", 125 => "01000100011010", 
    126 => "00100101110001", 127 => "00111000010101", 128 => "00111100011001", 
    129 => "00000010010101", 130 => "00011000011001", 131 => "11101010010101", 
    132 => "01010001000100", 133 => "11001010100100", 134 => "11000101111100", 
    135 => "11101010100110", 136 => "11001111111111", 137 => "00001111010000", 
    138 => "11110101110100", 139 => "00000010011001", 140 => "00000100001101", 
    141 => "11011011111111", 142 => "10110111001011", 143 => "01011111000100", 
    144 => "10111000011101", 145 => "11010110010101", 146 => "00011111101111", 
    147 => "10011101101111", 148 => "01001000101100", 149 => "11100110010000", 
    150 => "11011110011011", 151 => "01010011001101", 152 => "00110110011110", 
    153 => "11100110011011", 154 => "01000001110010", 155 => "01011010011110", 
    156 => "00101001110101", 157 => "10110110010111", 158 => "01011100000101", 
    159 => "11011100110000", 160 => "00011010111000", 161 => "11110000011010", 
    162 => "11011001100110", 163 => "11110000110101", 164 => "01001111110100", 
    165 => "00010101011001", 166 => "11000101111001", 167 => "01001101101000", 
    168 => "00100100001110", 169 => "01010001000110", 170 => "00101100111000", 
    171 => "01100001010111", 172 => "10010001011000", 173 => "10100111011000", 
    174 => "00100001101000", 175 => "00011001100101", 176 => "11100001010000", 
    177 => "11110011011011", 178 => "11101010010111", 179 => "01100000100011", 
    180 => "10111011000100", 181 => "01001011000100", 182 => "10110000000010", 
    183 => "10011111000000", 184 => "10011000111011", 185 => "00100011011111", 
    186 => "10011100110000", 187 => "01011010001000", 188 => "00010001110000", 
    189 => "00010100100110", 190 => "11011101011000", 191 => "00001011000110", 
    192 => "00110001101010", 193 => "00101010011001", 194 => "00111101110110", 
    195 => "10111001111111", 196 => "11100011000011", 197 => "00110011111101", 
    198 => "10100110100001", 199 => "01001011011010", 200 => "11110010000101", 
    201 => "11011101111000", 202 => "10110100011001", 203 => "00010100001001", 
    204 => "01000001010100", 205 => "10101110110101", 206 => "00010110110010", 
    207 => "10111110110000", 208 => "10010011000100", 209 => "00001010011101", 
    210 => "11011011011001", 211 => "00110111010101", 212 => "00011011110111", 
    213 => "11110100001001", 214 => "10101011100100", 215 => "00011110101110", 
    216 => "01000101100011", 217 => "00110001110100", 218 => "10100110100111", 
    219 => "00001101111000", 220 => "11011100000101", 221 => "10101001001000", 
    222 => "00110100111110", 223 => "11110001101000", 224 => "11001100000110", 
    225 => "01001100101000", 226 => "10111001001111", 227 => "10101110010111", 
    228 => "01100000000001", 229 => "11011110100010", 230 => "01010010110100", 
    231 => "10110001010111", 232 => "00010011100000", 233 => "11111000110000", 
    234 => "11001101001010", 235 => "00011010000001", 236 => "11100111110100", 
    237 => "00110010111110", 238 => "10101100000110", 239 => "10110101011101", 
    240 => "00010011000111", 241 => "00111001001010", 242 => "00101101000010", 
    243 => "11011001000010", 244 => "00101100000111", 245 => "01011100001001", 
    246 => "10101111000101", 247 => "00011110111011", 248 => "10011100000101", 
    249 => "01100110110110", 250 => "11000101100011", 251 => "11110010100110", 
    252 => "01011001111101", 253 => "00010101100101", 254 => "00010101110010", 
    255 => "00101001011001", 256 => "00110001111100", 257 => "01100110111101", 
    258 => "11111110111111", 259 => "11010101010010", 260 => "11001110011101", 
    261 => "10101011010111", 262 => "11110110000011", 263 => "10111110110000", 
    264 => "00000001000110", 265 => "00000001010100", 266 => "11100100001101", 
    267 => "10100100111100", 268 => "00011110001110", 269 => "00001100001001", 
    270 => "11100111110011", 271 => "10110000010100", 272 => "00111010010001", 
    273 => "00100010010011", 274 => "01001000100111", 275 => "01000001001101", 
    276 => "00010010011100", 277 => "11011101110111", 278 => "00101100000111", 
    279 => "01001100100010", 280 => "00110110001001", 281 => "01001011111000", 
    282 => "10111101110110", 283 => "01100011110011", 284 => "00110101100111", 
    285 => "00000101100000", 286 => "00000001101000", 287 => "10100010000000" );


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

entity infer_layer_6_weights_V_16 is
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

architecture arch of infer_layer_6_weights_V_16 is
    component infer_layer_6_weights_V_16_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_6_weights_V_16_rom_U :  component infer_layer_6_weights_V_16_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;

