-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_6_weights_V_27_rom is 
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


architecture rtl of infer_layer_6_weights_V_27_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "1111111010001110", 1 => "1111110010110011", 2 => "0001001111010100", 
    3 => "1110111000001101", 4 => "1111010111010100", 5 => "1110111001001110", 
    6 => "0001011010001110", 7 => "0001011110111100", 8 => "1110111100111010", 
    9 => "0000111010011111", 10 => "0000111111111001", 11 => "1110100000111000", 
    12 => "0000110001010100", 13 => "0001011010000111", 14 => "1110110010000001", 
    15 => "1110100000001110", 16 => "1110101010111000", 17 => "0000010011001011", 
    18 => "1111111111001001", 19 => "0000111001010000", 20 => "1110111011010100", 
    21 => "1111000110100011", 22 => "0000100010010110", 23 => "0000001111110111", 
    24 => "1101110100010010", 25 => "1111101110101011", 26 => "0001001001001010", 
    27 => "0001011101111001", 28 => "0000110111100010", 29 => "0000011100000001", 
    30 => "1110100100001001", 31 => "1110111110111001", 32 => "1110101100111111", 
    33 => "0001100111000011", 34 => "1110001111101001", 35 => "0001001101011101", 
    36 => "1110110110100011", 37 => "1110110010010001", 38 => "1101011111100101", 
    39 => "0000101111011000", 40 => "0001100100001010", 41 => "1111110100110101", 
    42 => "0001001001100011", 43 => "0001010111011101", 44 => "1110111000101011", 
    45 => "1111101010000000", 46 => "1111100111100100", 47 => "0000000011101101", 
    48 => "0000001110010000", 49 => "0000001100110011", 50 => "1111111100010111", 
    51 => "1111111010110111", 52 => "0000100000011000", 53 => "1111101011101111", 
    54 => "1100111111100100", 55 => "1111100000011011", 56 => "1011111001111111", 
    57 => "1111010100100000", 58 => "1110010100110111", 59 => "0000110000010000", 
    60 => "1110011011100000", 61 => "1110111101010001", 62 => "0000010001010100", 
    63 => "0000001110000010", 64 => "1111100000001011", 65 => "0001100101111101", 
    66 => "1100010110111101", 67 => "1110101100111100", 68 => "1111101001001100", 
    69 => "1111011101011000", 70 => "1111111100100001", 71 => "0000110011001010", 
    72 => "0001011000000001", 73 => "1110111111100001", 74 => "0000000101100001", 
    75 => "1110111010111011", 76 => "1100010101000000", 77 => "1110000010010011", 
    78 => "1110001111001010", 79 => "0001011011101101", 80 => "0001100101010100", 
    81 => "1111101001101110", 82 => "1111000010011111", 83 => "0001010000001001", 
    84 => "1111101111100101", 85 => "0001100010000111", 86 => "1110010000010010", 
    87 => "1101111000000111", 88 => "1110011000011110", 89 => "0001001010010000", 
    90 => "1110000110011011", 91 => "1111010010010011", 92 => "0000001001101000", 
    93 => "0000100111000110", 94 => "1111010010111010", 95 => "1111011000010010", 
    96 => "0000001010011110", 97 => "0000111100111001", 98 => "0010100100110011", 
    99 => "0000101110100101", 100 => "1110101110000001", 101 => "1110101000101000", 
    102 => "0000010010010011", 103 => "0001100000001000", 104 => "1111001111110010", 
    105 => "0000000101001110", 106 => "1111101100110010", 107 => "1111101110011100", 
    108 => "0000000101101010", 109 => "0010000101110000", 110 => "1111110011001100", 
    111 => "0001001100100100", 112 => "0001100110001010", 113 => "1111001010110011", 
    114 => "0001001100111111", 115 => "0000011110001111", 116 => "1111100010101101", 
    117 => "1110111111001000", 118 => "0000111000011100", 119 => "0010010010111000", 
    120 => "1011101111010010", 121 => "1111111110011110", 122 => "1111000110001000", 
    123 => "0001001000100001", 124 => "1110110000011011", 125 => "0001000100111101", 
    126 => "1111101110110111", 127 => "1111001000001111", 128 => "1110110010000001", 
    129 => "0001010000001010", 130 => "1111100100011111", 131 => "0000111010011010", 
    132 => "1111110011110001", 133 => "0000000100000100", 134 => "0001101100011001", 
    135 => "0001000111011001", 136 => "0000010000110100", 137 => "0001100101100011", 
    138 => "1111011011101011", 139 => "0000110001000110", 140 => "1111001110010011", 
    141 => "1111100110100001", 142 => "1110011111101000", 143 => "0000000001011100", 
    144 => "0000011010010000", 145 => "1111111101101100", 146 => "1111011011000010", 
    147 => "0000010010111111", 148 => "1111001100100100", 149 => "0001001011011010", 
    150 => "0000000001001100", 151 => "1111011111000111", 152 => "1101010000100010", 
    153 => "1111000110001000", 154 => "1110010000101101", 155 => "1110011110110000", 
    156 => "0000110001110010", 157 => "0000100000111001", 158 => "1111000001001010", 
    159 => "1110100111010111", 160 => "1111111111110010", 161 => "0000011001110000", 
    162 => "1111110100011111", 163 => "0000111111111011", 164 => "0001010111111110", 
    165 => "1111110111101001", 166 => "0000100100001001", 167 => "0001010010111000", 
    168 => "1111110010011111", 169 => "0000100001011011", 170 => "0001010000111001", 
    171 => "0001011011101011", 172 => "1110001110011011", 173 => "1101010001000011", 
    174 => "0000111011110000", 175 => "0001101110000100", 176 => "0001100111110100", 
    177 => "1111011000110100", 178 => "0000100000111100", 179 => "1111001100111101", 
    180 => "1111000110101100", 181 => "0000000001101011", 182 => "0000011001101111", 
    183 => "1101110110101011", 184 => "1101001000001001", 185 => "0000000111010000", 
    186 => "1110001011111011", 187 => "0001011000011010", 188 => "1110111100011110", 
    189 => "1111011001000000", 190 => "0000001010010111", 191 => "0000100111100001", 
    192 => "0000000000010010", 193 => "0000010001000100", 194 => "1111111110110110", 
    195 => "1111011010011101", 196 => "1111011001001000", 197 => "1110100101101100", 
    198 => "0000010100000000", 199 => "1110100000000101", 200 => "1110011110001101", 
    201 => "1111110110000101", 202 => "1110011111100110", 203 => "1111000011010011", 
    204 => "0010000110011100", 205 => "0001001101100000", 206 => "0001100001101001", 
    207 => "1110110000000101", 208 => "1110111100010111", 209 => "0001010110100110", 
    210 => "1111000001110000", 211 => "0001011100000000", 212 => "1110110101100101", 
    213 => "1110111100110011", 214 => "0010110100010010", 215 => "1110010010001001", 
    216 => "0000001000100111", 217 => "0000000101111000", 218 => "1111011100010000", 
    219 => "0000111110100001", 220 => "0000110101111101", 221 => "1111110000110100", 
    222 => "1111100010101100", 223 => "0000000101010000", 224 => "0010010001110000", 
    225 => "0000110111101110", 226 => "1110011101111100", 227 => "1110011110110110", 
    228 => "1111011011110000", 229 => "0000100100000101", 230 => "1111011001011001", 
    231 => "0001010111010010", 232 => "1111010110110000", 233 => "0000001101010100", 
    234 => "1111011110010101", 235 => "1110100101000000", 236 => "1110010101100100", 
    237 => "1110010100001111", 238 => "0001010011010011", 239 => "0000110011011101", 
    240 => "0010001110001111", 241 => "0000000000011000", 242 => "1111111100111001", 
    243 => "1111010010011000", 244 => "0000101010100101", 245 => "0000100000001011", 
    246 => "0011010001010001", 247 => "0011101110100100", 248 => "0011101111110110", 
    249 => "1110100011101011", 250 => "0001100111101100", 251 => "0001100001101001", 
    252 => "0001100010101100", 253 => "0000101100101101", 254 => "0000100001011001", 
    255 => "1110100101011011", 256 => "1111011110111100", 257 => "0000011100111101", 
    258 => "1101101101011101", 259 => "0001001101000001", 260 => "0001100101111110", 
    261 => "1111001111010101", 262 => "1111110010100011", 263 => "0001001101110010", 
    264 => "0000001001000001", 265 => "1111100101111110", 266 => "0001011100111101", 
    267 => "0000101111010010", 268 => "1110100010001100", 269 => "0000010000001011", 
    270 => "0000011001101100", 271 => "1111011110000010", 272 => "0010000011101110", 
    273 => "0010010010010110", 274 => "1110100100000011", 275 => "0001100011100100", 
    276 => "1111101101101101", 277 => "0001001101100110", 278 => "0001111011100011", 
    279 => "1111010110110110", 280 => "0001010100111111", 281 => "1111001000000011", 
    282 => "0010001010011011", 283 => "0000101110100011", 284 => "1111111001000100", 
    285 => "0001001011110110", 286 => "1111010110001100", 287 => "1111111011101101" );


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

entity infer_layer_6_weights_V_27 is
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

architecture arch of infer_layer_6_weights_V_27 is
    component infer_layer_6_weights_V_27_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_6_weights_V_27_rom_U :  component infer_layer_6_weights_V_27_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


