module mod_ReLU
use mod_Precision
use mod_BaseActivationFunction
implicit none    

!-------------------
! �����ࣺ����� |
!-------------------
type, extends(BaseActivationFunction), public :: ReLU
    !* �̳���BaseActivationFunction��ʵ����ӿ�

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f       => m_fun_ReLU
    procedure, public :: f_vect  => m_fun_ReLU_vect 
    procedure, public :: df      => m_df_ReLU
    procedure, public :: df_vect => m_df_ReLU_vect

end type ReLU
!===================

    !-------------------------
    private :: m_fun_ReLU
    private :: m_df_ReLU
    private :: m_fun_ReLU_vect
	private :: m_df_ReLU_vect
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* ReLU����
    subroutine m_fun_ReLU( this, index, x, y )
    implicit none
        class(ReLU), intent(inout) :: this
        integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: y
    
        if (x(index) < 0) then
            y = 0
        else
            y = x(index)
        end if
    
        return
    end subroutine
    !====
    
    !* �������������ReLU����
	subroutine m_fun_ReLU_vect( this, x, y )
	implicit none
        class(ReLU), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
	
		integer :: count, i
	
        count = SIZE(X)
        
        do i=1, count
            if (x(i) < 0) then
                y(i) = 0
            else
                y(i) = x(i)
            end if
        end do
        
		return
	end subroutine
	!====
    
	!* ReLU������һ�׵���
	subroutine m_df_ReLU( this, index, x, dy )
	implicit none
        class(ReLU), intent(inout) :: this
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: dy
	
		if (x(index) < 0) then
            dy = 0
        else
            dy = 1
        end if
	
		return
	end subroutine
	!====
	
	!* �������������ReLU������һ�׵���
	subroutine m_df_ReLU_vect( this, x, dy )
	implicit none
        class(ReLU), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy
	
		integer :: count, i
	
        count = SIZE(X)
        
        do i=1, count
            if (x(i) < 0) then
                dy(i) = 0
            else
                dy(i) = 1
            end if
        end do
	
		return
	end subroutine
	!====

end module