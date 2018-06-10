module mod_BaseActivationFunction
implicit none
    
!------------------
! 抽象类：激活函数 |
!------------------
type, abstract, public :: BaseActivationFunction

!||||||||||||    
contains   !|
!||||||||||||

    !* 激活函数
    procedure(m_f), deferred, public :: f 
    !* 接收向量参数的激活函数
    procedure(m_f_vect), deferred, public :: f_vect 
    !* 激活函数导数
    procedure(m_df), deferred, public :: df 
    !* 接收向量参数的激活函数导数
    procedure(m_df_vect), deferred, public :: df_vect  

end type BaseActivationFunction
!===================
    

!------------------
! 抽象类：激活函数 |
!------------------	
abstract interface   

	!* 激活函数
	subroutine m_f( this, x, y )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		real(PRECISION), intent(in) :: x
		real(PRECISION), intent(out) :: y

	end subroutine
	!====
    
    !* 接收向量参数的激活函数
	subroutine m_f_vect( this, x, y )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
		
	end subroutine
	!====
	
	!* 激活函数一阶导数
	subroutine m_df( this, x, dy )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		real(PRECISION), intent(in) :: x
		real(PRECISION), intent(out) :: dy

	end subroutine
	!====
	
	!* 接收向量参数的激活函数一阶导数
	subroutine m_df_vect( this, x, dy )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy

	end subroutine
	!====

end interface
!===================
    
end module