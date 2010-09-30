! Copyright 2010 The Fatiando a Terra Development Team
!
! This file is part of Fatiando a Terra.
!
! Fatiando a Terra is free software: you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! Fatiando a Terra is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Extention module for fatiando.heat.diffusionfd
!
! Functions:
!   * timestep1d: Perform a single time step of the 1D diffusion equation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



! Perform a single time step of the 1D diffusion equation.
! Parameters:
!   temp_t: 1D array with the current temperature on the FD nodes
!   diffusivity: 1D array with the thermal diffusivity on the FD nodes
!   nnodes: number of FD nodes
!   deltat: time interval between steps
!   deltax: x interval between FD nodes
! Return parameter:
!   temp_tp1: 1D array with the future temperature on the FD nodes
SUBROUTINE timestep1d(temp_t, diffusivity, nnodes, deltat, deltax, &
                              temp_tp1)

    IMPLICIT NONE

    INTEGER*4, INTENT(IN) :: nnodes
    REAL*8, INTENT(IN) :: deltat, deltax
    REAL*8, INTENT(IN) :: temp_t(nnodes), diffusivity(nnodes)
    REAL*8, INTENT(OUT) :: temp_tp1(nnodes)
    INTEGER*4 :: i

    DO i = 1, nnodes - 1

        temp_tp1(i) = (diffusivity(i)*deltat/(deltax**2))* &
                        (temp_t(i+1) - 2*temp_t(i) + temp_t(i-1)) + temp_t(i)

    ENDDO

END
