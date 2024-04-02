-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_10_weights_V_40_rom is 
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


architecture rtl of infer_layer_10_weights_V_40_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "0001001111101111", 1 => "0100101010011001", 2 => "0001000001011111", 
    3 => "0011011111100110", 4 => "0100000001111101", 5 => "0000100101011011", 
    6 => "0011111111001000", 7 => "0010101111000101", 8 => "1111101001011000", 
    9 => "1011101001110101", 10 => "1110010100011000", 11 => "1101100110111110", 
    12 => "1110100110110101", 13 => "1011110001110001", 14 => "0001001101000001", 
    15 => "1100101011111101", 16 => "0000011101000101", 17 => "1101010110101111", 
    18 => "0011011110011100", 19 => "0001001011011010", 20 => "0001011111000011", 
    21 => "1110101100010110", 22 => "1100000001001101", 23 => "1110001001100111", 
    24 => "0011000101010110", 25 => "1011011110010000", 26 => "0000110001001110", 
    27 => "1111111001110001", 28 => "1110110001010000", 29 => "1101110110100010", 
    30 => "0010100111001100", 31 => "0010011011110110" );


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

entity infer_layer_10_weights_V_40 is
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

architecture arch of infer_layer_10_weights_V_40 is
    component infer_layer_10_weights_V_40_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_10_weights_V_40_rom_U :  component infer_layer_10_weights_V_40_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


