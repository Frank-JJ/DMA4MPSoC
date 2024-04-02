-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_10_weights_V_25_rom is 
    generic(
             DWIDTH     : integer := 16; 
             AWIDTH     : integer := 5; 
             MEM_SIZE    : integer := 32
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of infer_layer_10_weights_V_25_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "0000101110010001", 1 => "0100001000111011", 2 => "1111100001010100", 
    3 => "1110101000111100", 4 => "1111011011100001", 5 => "1111011001111101", 
    6 => "0100110111010001", 7 => "1111000110000010", 8 => "1101001110011111", 
    9 => "1111010100110000", 10 => "0011000110111111", 11 => "0100100001101010", 
    12 => "0001010010100000", 13 => "1100111011101001", 14 => "0011111010101100", 
    15 => "1111100010000100", 16 => "1100101101110111", 17 => "1111101001100101", 
    18 => "1100001100110101", 19 => "0000010111011000", 20 => "1101000010000101", 
    21 => "0010001111011101", 22 => "0011100000001111", 23 => "1111011100000101", 
    24 => "0011011011010001", 25 => "1111011001111001", 26 => "0100100010010101", 
    27 => "0011001111001011", 28 => "1110101111110100", 29 => "0000111011111011", 
    30 => "1101110100100001", 31 => "1110111000111110" );


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

entity infer_layer_10_weights_V_25 is
    generic (
        DataWidth : INTEGER := 16;
        AddressRange : INTEGER := 32;
        AddressWidth : INTEGER := 5);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of infer_layer_10_weights_V_25 is
    component infer_layer_10_weights_V_25_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_10_weights_V_25_rom_U :  component infer_layer_10_weights_V_25_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


