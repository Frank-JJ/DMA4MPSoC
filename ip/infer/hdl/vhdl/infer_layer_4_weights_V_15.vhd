-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_4_weights_V_15_rom is 
    generic(
             DWIDTH     : integer := 16; 
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


architecture rtl of infer_layer_4_weights_V_15_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "1110101011010101", 1 => "0001010010000110", 2 => "1110011011010001", 
    3 => "0000010010110110", 4 => "1111000001101110", 5 => "0000010001110111", 
    6 => "0000001100010010", 7 => "1110110000011111", 8 => "1111101000010111", 
    9 => "1110100001100010", 10 => "0001000001100010", 11 => "0000101101111101", 
    12 => "0001100010111110", 13 => "0001000110111101", 14 => "1110101001100101", 
    15 => "1111001110000000", 16 => "1110011000001000", 17 => "0001000111101001", 
    18 => "0000101000011100", 19 => "1111011011010110", 20 => "1111111001111011", 
    21 => "1110011101100011", 22 => "1111001111101110", 23 => "1111000010011110", 
    24 => "0000010101010100", 25 => "0000010000000111", 26 => "0000110100000111", 
    27 => "1111010001110001", 28 => "0000010011110110", 29 => "0000110111000010", 
    30 => "1111100011000111", 31 => "1111010011001001", 32 => "0000010101101110", 
    33 => "0000010111000100", 34 => "1111010111111101", 35 => "1111001010101101", 
    36 => "1111101100000101", 37 => "0000101000101100", 38 => "0000010010000110", 
    39 => "1111000000111101", 40 => "1110101010101001", 41 => "1111010111111110", 
    42 => "1110011001110001", 43 => "0000100001111111", 44 => "0000101010001100", 
    45 => "1110010110000000", 46 => "0000010001011110", 47 => "1111010101000100", 
    48 => "1110011111011010", 49 => "0000100001010110", 50 => "0000011010001011", 
    51 => "1111110011001110", 52 => "1111001100111010", 53 => "0001001001011011", 
    54 => "0000000111011100", 55 => "1110110111000110", 56 => "0001001011110100", 
    57 => "1110101111100111", 58 => "1110100010011010", 59 => "0001010010111110", 
    60 => "1111101111100001", 61 => "0000111101111000", 62 => "0000001111010011", 
    63 => "1111010101110001", 64 => "0001000001001101", 65 => "1111011110010011", 
    66 => "1111111111101001", 67 => "1111110100100111", 68 => "0000101100001011", 
    69 => "1111111001101111", 70 => "0000001001101011", 71 => "0001000011010011", 
    72 => "0000001011111111", 73 => "0000010100101001", 74 => "0000111000010111", 
    75 => "0000101101000111", 76 => "0000000110010011", 77 => "1111110000100001", 
    78 => "0001000101011111", 79 => "1111110000000000", 80 => "1111001010110011", 
    81 => "0000000100100000", 82 => "0011001110100100", 83 => "0000011110000110", 
    84 => "0001011011111001", 85 => "1111001000111011", 86 => "1111001000100001", 
    87 => "1110101000001011", 88 => "1110110101001000", 89 => "1111011000110111", 
    90 => "1111101001111001", 91 => "0001000111010100", 92 => "1110001000000110", 
    93 => "0001010001000010", 94 => "1111001011001001", 95 => "0000111101111010", 
    96 => "1111100001111110", 97 => "1111111111001110", 98 => "0001100111010011", 
    99 => "1110100001110011", 100 => "0001000011100100", 101 => "1111110100111100", 
    102 => "0000011100011111", 103 => "0000110100101010", 104 => "1110100111100001", 
    105 => "1111100101100111", 106 => "1111000001011000", 107 => "1111001011100101", 
    108 => "1110100100010010", 109 => "1111001110011011", 110 => "0000100111111110", 
    111 => "0000010101011001", 112 => "0000001100011001", 113 => "1111111111010111", 
    114 => "0010000110001101", 115 => "0001101110001001", 116 => "0001101011001100", 
    117 => "1111011100101110", 118 => "0001010000011111", 119 => "1111001011000000", 
    120 => "0000110011000001", 121 => "0000111101011110", 122 => "1110101011001101", 
    123 => "1110101011110111", 124 => "1111100000010101", 125 => "1110011110001101", 
    126 => "0000000011111100", 127 => "1111010100000010", 128 => "1101110001000011", 
    129 => "1111000011010111", 130 => "1110100110101111", 131 => "0000011001100001", 
    132 => "0000000101110101", 133 => "0000110001111011", 134 => "1111011110000000", 
    135 => "1111101110101001", 136 => "0000111000001110", 137 => "0001100001110111", 
    138 => "0001011101000010", 139 => "0001000000011001", 140 => "0001011101001100", 
    141 => "1110010100010110", 142 => "0001100000000100", 143 => "0000100110111100", 
    144 => "0001011011011000", 145 => "1111111000000011", 146 => "0010000101010000", 
    147 => "0001010110000000", 148 => "1111110010000011", 149 => "1111011111001000", 
    150 => "0000010111111100", 151 => "0001011001001000", 152 => "1111111101001010", 
    153 => "1110110100101011", 154 => "1111001011110111", 155 => "1111110000110101", 
    156 => "1101110100100101", 157 => "1111000101100111", 158 => "1101011011110101", 
    159 => "1111011111010110", 160 => "1110110000110101", 161 => "0000100011001110", 
    162 => "0001100100110111", 163 => "0001000100000100", 164 => "1111111110011000", 
    165 => "0001010110001111", 166 => "0001010110100101", 167 => "1111100110000100", 
    168 => "0001001100110000", 169 => "0000000011111111", 170 => "0000110101011111", 
    171 => "1110011101000011", 172 => "0000000101100001", 173 => "0000100110110111", 
    174 => "0001100100001000", 175 => "1111100011000101", 176 => "0000000001100100", 
    177 => "0000111100000111", 178 => "0001110110001010", 179 => "0000011001000000", 
    180 => "0000101001100111", 181 => "0000011101101101", 182 => "0000101101101001", 
    183 => "1111100011000011", 184 => "1110100010111111", 185 => "0000101101001111", 
    186 => "1111101011000100", 187 => "1110100011100111", 188 => "1111110101101010", 
    189 => "0001001110101101", 190 => "0010010110110100", 191 => "1111010110111010", 
    192 => "1101001101111100", 193 => "1111100001111111", 194 => "1111001011100000", 
    195 => "0000011000010110", 196 => "1111000100110100", 197 => "1111100000000011", 
    198 => "0000101011101011", 199 => "1111001101111010", 200 => "1111010001101011", 
    201 => "1111101110000011", 202 => "1111000011011110", 203 => "1111001000011011", 
    204 => "1111011101111101", 205 => "1110010110100101", 206 => "1110010111111110", 
    207 => "0001100001010001", 208 => "0000011011010011", 209 => "0001001001110001", 
    210 => "0101110011000110", 211 => "0001110010001100", 212 => "0000001100110011", 
    213 => "0000110100110010", 214 => "0001010011111010", 215 => "0000100101000000", 
    216 => "0001000001001101", 217 => "1100111110011100", 218 => "0000001001101001", 
    219 => "1110010111101101", 220 => "1011011110011101", 221 => "0000100111101011", 
    222 => "1101101000011100", 223 => "1110000000011100", 224 => "1111001111101100", 
    225 => "1110011001000000", 226 => "1111101110111111", 227 => "1111101010010111", 
    228 => "0000111011011010", 229 => "1110111101011111", 230 => "0000111110100110", 
    231 => "1111111110110011", 232 => "1110010101110100", 233 => "1110100110100010", 
    234 => "0000101011011101", 235 => "0001011101100001", 236 => "1110101100011010", 
    237 => "1110101010010001", 238 => "1111101111111101", 239 => "0000100010111110", 
    240 => "0001100100100001", 241 => "0001000010110101", 242 => "0100101110001001", 
    243 => "1101111110000111", 244 => "1110010001000110", 245 => "1111011010100001", 
    246 => "1111011001110100", 247 => "1111010000111111", 248 => "0000100010101000", 
    249 => "1110100000110010", 250 => "1111011001110100", 251 => "1111111001100101", 
    252 => "1111001110000001", 253 => "0001001001100011", 254 => "1110000111110101", 
    255 => "1101010101011110", 256 => "0000111010001010", 257 => "0000000001110010", 
    258 => "0001100010011101", 259 => "1111110000111010", 260 => "1110100000111100", 
    261 => "1110101001111001", 262 => "0000111000100010", 263 => "0000101111000110", 
    264 => "1111100110111011", 265 => "0000100010110000", 266 => "0000111000110111", 
    267 => "1111010110111100", 268 => "1110110001011010", 269 => "1111011110101101", 
    270 => "1110100010111111", 271 => "0001010101000100", 272 => "0000111001011000", 
    273 => "0001100010110001", 274 => "0000001110111000", 275 => "1100010011000100", 
    276 => "0000100100100000", 277 => "0000100110010101", 278 => "1110011010010001", 
    279 => "0000011010100011", 280 => "1110100011100111", 281 => "0010001011000000", 
    282 => "1110011010110111", 283 => "1110110110100101", 284 => "1111001001011010", 
    285 => "0000111110110110", 286 => "0000011100110110", 287 => "1111100101010110" );


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

entity infer_layer_4_weights_V_15 is
    generic (
        DataWidth : INTEGER := 16;
        AddressRange : INTEGER := 288;
        AddressWidth : INTEGER := 9);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of infer_layer_4_weights_V_15 is
    component infer_layer_4_weights_V_15_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_4_weights_V_15_rom_U :  component infer_layer_4_weights_V_15_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


