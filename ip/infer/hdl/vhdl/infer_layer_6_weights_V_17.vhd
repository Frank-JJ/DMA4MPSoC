-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_6_weights_V_17_rom is 
    generic(
             DWIDTH     : integer := 15; 
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


architecture rtl of infer_layer_6_weights_V_17_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "111100101110010", 1 => "111110100000101", 2 => "111011110111111", 
    3 => "001000000110111", 4 => "110111111001010", 5 => "001011100000011", 
    6 => "111111111110111", 7 => "110101001110001", 8 => "110110111110011", 
    9 => "111101100000000", 10 => "001011011101000", 11 => "111011100100000", 
    12 => "000001010111001", 13 => "110100111111100", 14 => "000111000000110", 
    15 => "101111011001000", 16 => "110010000110000", 17 => "000011111101010", 
    18 => "111001100011110", 19 => "000100001001011", 20 => "001001101110000", 
    21 => "111110101101010", 22 => "111101110111001", 23 => "000010001010000", 
    24 => "111000110010101", 25 => "000010000011110", 26 => "111010010110111", 
    27 => "111010111000011", 28 => "000001000010001", 29 => "000000101101100", 
    30 => "111000110110101", 31 => "000000110101100", 32 => "110110110001110", 
    33 => "110110000010011", 34 => "000111110000110", 35 => "000000110111100", 
    36 => "001011100011000", 37 => "110110011011000", 38 => "000101010001110", 
    39 => "000101010111011", 40 => "001010100110000", 41 => "001000110010011", 
    42 => "111101110001111", 43 => "110011000000111", 44 => "000000001111011", 
    45 => "000001010100001", 46 => "000011100000110", 47 => "111101010110101", 
    48 => "001000011001110", 49 => "000000000110011", 50 => "111001001110101", 
    51 => "111101111111011", 52 => "110111001010101", 53 => "111000110010000", 
    54 => "000100000001001", 55 => "000010111010111", 56 => "111010000001001", 
    57 => "110100010101111", 58 => "110010110011100", 59 => "001100001100000", 
    60 => "001100001001001", 61 => "001010011010010", 62 => "110011111010110", 
    63 => "001000001111111", 64 => "110011001001011", 65 => "111101101011001", 
    66 => "111110111000101", 67 => "111111100101100", 68 => "111101001110011", 
    69 => "111000011111110", 70 => "111011001001010", 71 => "000110010011001", 
    72 => "111100101001001", 73 => "110100011010111", 74 => "000110100101110", 
    75 => "110011001010110", 76 => "000101111001001", 77 => "000100001110100", 
    78 => "000101010000111", 79 => "000100011011001", 80 => "110111001001001", 
    81 => "000010110010000", 82 => "111010010000110", 83 => "000001010110111", 
    84 => "111001110101011", 85 => "110100111010111", 86 => "001000011100111", 
    87 => "110001011011001", 88 => "111110011100011", 89 => "111010110111110", 
    90 => "110001111101010", 91 => "111110101100110", 92 => "111011110100001", 
    93 => "111101000001100", 94 => "000011010001011", 95 => "110111101011110", 
    96 => "000101000101010", 97 => "001010110000000", 98 => "000000001010110", 
    99 => "111111100001000", 100 => "001010100110111", 101 => "110011011010010", 
    102 => "000001110010100", 103 => "111101011010110", 104 => "111101011110011", 
    105 => "000110110001001", 106 => "110100001010000", 107 => "111100000101001", 
    108 => "000110100001010", 109 => "111110100010111", 110 => "000101010011100", 
    111 => "000111000000111", 112 => "111001000001100", 113 => "001001001110110", 
    114 => "111110001100011", 115 => "111110110000010", 116 => "000101000001001", 
    117 => "110111001010000", 118 => "000010010100010", 119 => "110110110100101", 
    120 => "110001010010100", 121 => "110111001011101", 122 => "110111011000011", 
    123 => "110011111010011", 124 => "111011111010100", 125 => "110111000000101", 
    126 => "111100010100111", 127 => "000111010101011", 128 => "111001101110110", 
    129 => "000011011010000", 130 => "111001011011100", 131 => "001001110010001", 
    132 => "000101010101110", 133 => "110111000010101", 134 => "110011011001100", 
    135 => "110111101110101", 136 => "000000000011010", 137 => "000110011101100", 
    138 => "111000011011011", 139 => "111100101001101", 140 => "000001010010100", 
    141 => "111001011011010", 142 => "111000101100011", 143 => "110010101101111", 
    144 => "110011100100110", 145 => "111011110111001", 146 => "110110010101000", 
    147 => "000100110100101", 148 => "000001011100101", 149 => "000010111101001", 
    150 => "111111011110010", 151 => "110011111110111", 152 => "000010110011010", 
    153 => "111001000100001", 154 => "000001101111011", 155 => "110100111110100", 
    156 => "110101011001110", 157 => "110100101100110", 158 => "000001011110100", 
    159 => "001010111100010", 160 => "000100100110101", 161 => "000001110000100", 
    162 => "000111111001101", 163 => "000010000010100", 164 => "111101001110111", 
    165 => "001001001010100", 166 => "001001010000010", 167 => "001010100100000", 
    168 => "110111010000010", 169 => "000100110100100", 170 => "111011001001100", 
    171 => "110101010111100", 172 => "111001111011001", 173 => "110101111101101", 
    174 => "111011000100110", 175 => "000010001101111", 176 => "110111001111010", 
    177 => "111011001111110", 178 => "111000011100101", 179 => "000000001000010", 
    180 => "001001000011110", 181 => "111001101101100", 182 => "110010100100110", 
    183 => "110010110111001", 184 => "000001010111100", 185 => "111001100000101", 
    186 => "110011000000010", 187 => "000100101111010", 188 => "000011001010001", 
    189 => "111111010100101", 190 => "001100111101100", 191 => "000001001100000", 
    192 => "001000010101000", 193 => "001100000011001", 194 => "110001101101100", 
    195 => "111011011011010", 196 => "001010011100011", 197 => "000001101100100", 
    198 => "000110101001111", 199 => "110111011000010", 200 => "001100100110111", 
    201 => "000011011100101", 202 => "001100111001110", 203 => "110110110101000", 
    204 => "110000110011001", 205 => "111001011011001", 206 => "111010001110101", 
    207 => "000000000100110", 208 => "110101111101110", 209 => "110011010100100", 
    210 => "111110001000010", 211 => "111100100100010", 212 => "111100100001001", 
    213 => "111001011110001", 214 => "111110000000111", 215 => "110001001100101", 
    216 => "110000110111001", 217 => "111100111101110", 218 => "001010011011010", 
    219 => "001010101100100", 220 => "000011010001001", 221 => "000111000011111", 
    222 => "000110111110110", 223 => "110110101101000", 224 => "110111101011010", 
    225 => "000000001111010", 226 => "110101111111100", 227 => "001011110000000", 
    228 => "000110011011000", 229 => "001011110110010", 230 => "000101100101100", 
    231 => "001010101011111", 232 => "000111001111010", 233 => "001100101001101", 
    234 => "000111111001000", 235 => "000010100101010", 236 => "000001010100100", 
    237 => "110101110111001", 238 => "001001011000110", 239 => "000000001101000", 
    240 => "000011111100110", 241 => "110100110110001", 242 => "001001110001111", 
    243 => "111110010011101", 244 => "111011100011110", 245 => "000100001111001", 
    246 => "000111010011111", 247 => "000010001111010", 248 => "111001111010001", 
    249 => "000111011001000", 250 => "111100101010010", 251 => "000010101001010", 
    252 => "001011101110111", 253 => "110011001100001", 254 => "110110100101110", 
    255 => "000111111111111", 256 => "110111011001011", 257 => "000011000100011", 
    258 => "000110101101111", 259 => "110111110100000", 260 => "110110100100100", 
    261 => "110110001000101", 262 => "110011101101100", 263 => "110011000110000", 
    264 => "111001011010001", 265 => "000111111101001", 266 => "000001001011110", 
    267 => "000101100000001", 268 => "000011001100000", 269 => "000110111110010", 
    270 => "110010111110111", 271 => "000001111000101", 272 => "110100001101010", 
    273 => "000000001111001", 274 => "000010100010010", 275 => "000000111011001", 
    276 => "000010010000110", 277 => "000000000001110", 278 => "110011010001011", 
    279 => "000110000111101", 280 => "000011100001101", 281 => "000010001001111", 
    282 => "110101101101110", 283 => "001000010001001", 284 => "000010111010100", 
    285 => "111111101110001", 286 => "110101110110010", 287 => "001000000110100" );


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

entity infer_layer_6_weights_V_17 is
    generic (
        DataWidth : INTEGER := 15;
        AddressRange : INTEGER := 288;
        AddressWidth : INTEGER := 9);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of infer_layer_6_weights_V_17 is
    component infer_layer_6_weights_V_17_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_6_weights_V_17_rom_U :  component infer_layer_6_weights_V_17_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


