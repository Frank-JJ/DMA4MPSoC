-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_4_weights_V_22_rom is 
    generic(
             DWIDTH     : integer := 17; 
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


architecture rtl of infer_layer_4_weights_V_22_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "00000110101110010", 1 => "00001100001101011", 
    2 => "11111010111001011", 3 => "00000111010010001", 
    4 => "11110011000010011", 5 => "00001111101101101", 
    6 => "00000001011111001", 7 => "00001001011011001", 
    8 => "00001001000000000", 9 => "11111001111011001", 
    10 => "11111100110001110", 11 => "11110101101000101", 
    12 => "00001010101101001", 13 => "00000001010111011", 
    14 => "00001001100111000", 15 => "11111110110110000", 
    16 => "11111010100001011", 17 => "00000101011010001", 
    18 => "00000001011100001", 19 => "11111101010111010", 
    20 => "00000000101111100", 21 => "00001110011110111", 
    22 => "11111001110101101", 23 => "11110110010110101", 
    24 => "11110100111001010", 25 => "00010001110011000", 
    26 => "11110111001010001", 27 => "11110100001111111", 
    28 => "00000000011101011", 29 => "11111111011111001", 
    30 => "00000110001100101", 31 => "11111110001111101", 
    32 => "11110011100001010", 33 => "11111111100111010", 
    34 => "00000111010001111", 35 => "00000100100100101", 
    36 => "11111110101011111", 37 => "11110100111111110", 
    38 => "00000111101000000", 39 => "11111000110110110", 
    40 => "00000100101011001", 41 => "00001000000010110", 
    42 => "00001000010101100", 43 => "00001010101001010", 
    44 => "00000100100011010", 45 => "00001100001101001", 
    46 => "11111011101011011", 47 => "11111100110100001", 
    48 => "11110101000010101", 49 => "11110110000000101", 
    50 => "00110010001000111", 51 => "00101011100110001", 
    52 => "00001011011100011", 53 => "11110100101110001", 
    54 => "11110110100000111", 55 => "00001000010001110", 
    56 => "00000100010011011", 57 => "11101110100011100", 
    58 => "00000110011111100", 59 => "11111110010111110", 
    60 => "11101110010100010", 61 => "00000000110100011", 
    62 => "11100110001100100", 63 => "00001111010110001", 
    64 => "11101011010010111", 65 => "00001010111110010", 
    66 => "00000011000110100", 67 => "11111010100000001", 
    68 => "00000111011111001", 69 => "11101010011101010", 
    70 => "11111101010000100", 71 => "00000010001011010", 
    72 => "11111011111100001", 73 => "00000110111100010", 
    74 => "11111100010110100", 75 => "11111111111000101", 
    76 => "11111111110011000", 77 => "11110001110010100", 
    78 => "00000001111000000", 79 => "11111111010000111", 
    80 => "11111011111001110", 81 => "11110010011011001", 
    82 => "00111010110010110", 83 => "00100000011001110", 
    84 => "11111011010111011", 85 => "11110011111010011", 
    86 => "00000110110110111", 87 => "00000100100110001", 
    88 => "00000000010100101", 89 => "11011111011011000", 
    90 => "00000011111001000", 91 => "00001010000110001", 
    92 => "11011010111010001", 93 => "11111011101100111", 
    94 => "11111001101011000", 95 => "11101100100101101", 
    96 => "00010100100100001", 97 => "00000001100001001", 
    98 => "11110100101001011", 99 => "00001010010100111", 
    100 => "00000000001110110", 101 => "00000111100110000", 
    102 => "11110100011010001", 103 => "11111101110000001", 
    104 => "00000001010000111", 105 => "11111001011111110", 
    106 => "00000000101110000", 107 => "11111110010000010", 
    108 => "00000001110110110", 109 => "00001011000010100", 
    110 => "11111010011111101", 111 => "00001000001010010", 
    112 => "00001010001011011", 113 => "00000000000011010", 
    114 => "11111000111110000", 115 => "00001110000011011", 
    116 => "11111101101111001", 117 => "00000110100111101", 
    118 => "11111100110000110", 119 => "11111101001101000", 
    120 => "11111110101101100", 121 => "00010111001111011", 
    122 => "11111111011111010", 123 => "11111110010110010", 
    124 => "00100100001010111", 125 => "11111101010100001", 
    126 => "00011101000101100", 127 => "00011001001010010", 
    128 => "00011001010010100", 129 => "11111100001011101", 
    130 => "11111110101110010", 131 => "00000110001011001", 
    132 => "00001000010110111", 133 => "11111011100101001", 
    134 => "00001011000010100", 135 => "11110111101010110", 
    136 => "11111001101111011", 137 => "00001010110000111", 
    138 => "00000011001101011", 139 => "11111111001010110", 
    140 => "00000001001011001", 141 => "00000111111011110", 
    142 => "00000001011011000", 143 => "11111000011010001", 
    144 => "11111101010010000", 145 => "11111101111001110", 
    146 => "00110000100100111", 147 => "00101100100010011", 
    148 => "11111101101111011", 149 => "11111001110110101", 
    150 => "11111100100011001", 151 => "00001011100100011", 
    152 => "00000001110101101", 153 => "11110101001010110", 
    154 => "00000101010010000", 155 => "00001100111011100", 
    156 => "00100111111111000", 157 => "11101110100101100", 
    158 => "00010100110010100", 159 => "00100100110110100", 
    160 => "00010101011101100", 161 => "11110001101010010", 
    162 => "00000100010010101", 163 => "11111011010000001", 
    164 => "00000011000000001", 165 => "00001010111111000", 
    166 => "11110011100110110", 167 => "11110011101010100", 
    168 => "00000000101010111", 169 => "11110011001001011", 
    170 => "11111110100000101", 171 => "11110100010111111", 
    172 => "00001011001001100", 173 => "11101100111100110", 
    174 => "00000110010111111", 175 => "11101101111101010", 
    176 => "11110101010010011", 177 => "00000000111011010", 
    178 => "00110110001011001", 179 => "00100011011001010", 
    180 => "11110010000000010", 181 => "11110010010010110", 
    182 => "00000101101000011", 183 => "00001100011001101", 
    184 => "11110111101000000", 185 => "11111101001000000", 
    186 => "00001001110110001", 187 => "00000001001110111", 
    188 => "00110011100111111", 189 => "11111010011111010", 
    190 => "00011111010100110", 191 => "00101010001001100", 
    192 => "00010111011000001", 193 => "00001101100000110", 
    194 => "00000010100000110", 195 => "11111010111000100", 
    196 => "00000011011101110", 197 => "00010000110110001", 
    198 => "11111010111111001", 199 => "11111001011000001", 
    200 => "00000110000010000", 201 => "00000101010101010", 
    202 => "11111010101000011", 203 => "11111001110110101", 
    204 => "11111101111101011", 205 => "00000011001100100", 
    206 => "11111000001111100", 207 => "00001011001110110", 
    208 => "11110011101010000", 209 => "00010000010000111", 
    210 => "11110101100011101", 211 => "11111101001110000", 
    212 => "00010010010111110", 213 => "00001111101111001", 
    214 => "11111111010010100", 215 => "11111100111001111", 
    216 => "00000010000001010", 217 => "00001111101100000", 
    218 => "11110011111010001", 219 => "00000101011111000", 
    220 => "00100111111111011", 221 => "11111110110010110", 
    222 => "00010010001011000", 223 => "00001100100001000", 
    224 => "00100111100011111", 225 => "00001000100110001", 
    226 => "00001001011001110", 227 => "00001110110001101", 
    228 => "11111101001110100", 229 => "00000110110101011", 
    230 => "00000011110001101", 231 => "00000110111111001", 
    232 => "11111011000001011", 233 => "00000110010101011", 
    234 => "11111010110110101", 235 => "00000011101101001", 
    236 => "00000101101111011", 237 => "11111110111111001", 
    238 => "11111100110011011", 239 => "11110110101111111", 
    240 => "00000110010011010", 241 => "00000010111011110", 
    242 => "00001110111101100", 243 => "00001101100110001", 
    244 => "00001011011010000", 245 => "00000111101111110", 
    246 => "11111100111111000", 247 => "00000000101111110", 
    248 => "11111101011101101", 249 => "11111000001111011", 
    250 => "00001001011001001", 251 => "00000110000001101", 
    252 => "00101110000010000", 253 => "11110110101000011", 
    254 => "00000010000010101", 255 => "00100111111110101", 
    256 => "00011111010010100", 257 => "11111100000011110", 
    258 => "11111010100010011", 259 => "11110000101011001", 
    260 => "00001011000101111", 261 => "00000111011110110", 
    262 => "00000111111101010", 263 => "00000000101001100", 
    264 => "11110011111101001", 265 => "11111100111001111", 
    266 => "00001011001010011", 267 => "11110100000011010", 
    268 => "00001000101100011", 269 => "00000010000001100", 
    270 => "11111010011000000", 271 => "11110001001011011", 
    272 => "00000000000111011", 273 => "11111111101000000", 
    274 => "00010000010000001", 275 => "00010101000010010", 
    276 => "11111100001001101", 277 => "11110110010000001", 
    278 => "00001011101000011", 279 => "00001100011110010", 
    280 => "00000000001001111", 281 => "00100001110100101", 
    282 => "00000100010111100", 283 => "00000110111000110", 
    284 => "01000000100001110", 285 => "11110111100001101", 
    286 => "00110001011101011", 287 => "00011010001110010" );


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

entity infer_layer_4_weights_V_22 is
    generic (
        DataWidth : INTEGER := 17;
        AddressRange : INTEGER := 288;
        AddressWidth : INTEGER := 9);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of infer_layer_4_weights_V_22 is
    component infer_layer_4_weights_V_22_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_4_weights_V_22_rom_U :  component infer_layer_4_weights_V_22_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


